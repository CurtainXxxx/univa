"""
UniVA 核心 Agent 编排层
=======================
定义三个 Agent 与总控系统：

  - PlanAgent   ：规划 Agent（无工具，纯 LLM 把请求拆成 JSON 执行计划）
  - ActAgent    ：执行 Agent（绑 MCP 工具，按计划逐步调用工具）
  - SingleAgent ：结论2 对照组（一个 Agent 同时规划+执行）
  - PlanActSystem：总装配（MCP 连接 + 记忆存储 + 对外 execute_task/execute_single）

调用链：用户输入 → PlanActSystem.execute_task
          → PlanAgent.generate_plan（拆计划）
          → ActAgent.execute_plan（执行每步）
          → inject_execution_results（记忆回写 SQLite，供下次规划）

本文件不直接调生成 API，真正的视频生成在 mcp_tools/*.py。
"""
import asyncio
import json
import os
import re
import traceback
from pathlib import Path
from dotenv import load_dotenv

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from agno.agent import Agent
from agno.tools.mcp import MCPTools, MultiMCPTools
from agno.db.sqlite import SqliteDb

# agno 2.8 默认把 system role 映射为 developer（为 GPT-5 设计），
# DeepSeek/DashScope 等 OpenAI 兼容 API 不认 developer role → 改回 system
from agno.models.openai.chat import OpenAIChat
OpenAIChat.default_role_map["system"] = "system"


def _init_env():
    base = Path(__file__).resolve().parents[1]
    env_file = base / ".env"
    if not env_file.exists():
        raise RuntimeError("Config missing: please copy univa/.env.example to univa/.env and fill your keys.")
    load_dotenv(dotenv_path=str(env_file), override=False)


_init_env()

from univa.config.config import config
from univa.utils.model_factory import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    prompt_dir = config.get('prompt_dir')
    prompt_path = os.path.join(prompt_dir, f"{prompt_name}.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_todo_progress_event(plan_data: Dict) -> Dict:
    if not plan_data or "execution_plan" not in plan_data:
        return None

    steps = plan_data["execution_plan"]["steps"]
    todo_items = []

    for i, step in enumerate(steps):
        status = step.get("status", "pending")
        description = step.get("action_description", f"Step {i+1}")

        if status == "success":
            todo_status = "completed"
        elif status == "ongoing":
            todo_status = "in_progress"
        else:
            todo_status = "pending"

        todo_items.append({
            "id": i,
            "description": description,
            "status": todo_status,
            "tool": step.get("tool", {}),
            "output": step.get("output", "")
        })

    return {
        "type": "todo_progress",
        "items": todo_items,
        "overall_description": plan_data["execution_plan"].get("overall_description", "")
    }


def extract_plan_from_content(content: str) -> Optional[Dict]:
    """从 LLM 输出里提取执行计划 JSON（支持 ```json 代码块或裸 JSON）。

    返回 dict（含 execution_plan）或原字符串（解析失败时）。
    """
    if not content or not isinstance(content, str):
        return content
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
            else:
                return content

        parsed_content = json.loads(json_str)
        if "execution_plan" in parsed_content:
            return parsed_content
        return content
    except (json.JSONDecodeError, Exception):
        return content


class PlanAgent:
    """规划 Agent：把用户请求拆成 JSON 执行计划。

    关键设计（论文 Plan-Act 的核心）：**不绑定任何工具**——
    规划阶段只负责"想清楚做什么"，真正调用工具的是 ActAgent。
    """

    def __init__(self, mcp_tools: MultiMCPTools, plan_db):
        self.mcp_tools = mcp_tools

        # 指令 = plan.txt（工具清单 + 计划 JSON 格式约束），指导 LLM 如何拆计划
        plan_prompt = load_prompt("plan")
        full_instructions = f"{plan_prompt}\n"

        # 模型配置从 univa/config/config.toml 读取（plan_model_* 段）
        plan_model_provider = config.get('plan_model_provider', 'openai')
        plan_model_id = config.get('plan_model_id', 'gpt-5-2025-08-07')
        plan_model_api_key = config.get('plan_model_api_key', '')
        plan_model_base_url = config.get('plan_model_base_url', '')
        plan_model_extra_params = config.get('plan_model_extra_params', '')

        self.agent = Agent(
            name="Univideo Plan Agent",
            model=create_model(
                provider=plan_model_provider,
                model_id=plan_model_id,
                api_key=plan_model_api_key,
                base_url=plan_model_base_url or None,
                extra_params=plan_model_extra_params,
            ),
            # 注意：没有 tools= 参数 → 规划 Agent 无法调用工具
            instructions=full_instructions,
            db=plan_db,                        # Task 记忆存这里（SQLite）
            add_history_to_context=True,       # 把历史消息拼进上下文
            num_history_messages=10,
            # session_state 是 agno 的会话级状态，execution_history 就是 Task Memory
            session_state={
                "execution_history": []
            }
        )

    def _get_available_tools_description(self) -> str:
        tools_info = []
        if hasattr(self.mcp_tools, 'tools') and self.mcp_tools.tools:
            for tool in self.mcp_tools.tools:
                tools_info.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_info) if tools_info else "No tools available"

    def extract_plan_from_content(self, content: str) -> Optional[Dict]:
        return extract_plan_from_content(content)

    async def generate_plan(self, session_id, user_request: str) -> Optional[Dict]:
        """生成执行计划。

        Task 记忆读取：从 SQLite 里取出本 session 的历史执行记录（execution_history），
        拼进输入上下文，让模型"记得上一轮做了什么"——这是论文 Task Memory 的读路径。
        """
        input_context = f"User Request: {user_request}\n"

        # 读记忆：拿到之前所有 {plan, execution_results}
        try:
            execution_historys = self.agent.get_session_state(session_id).get("execution_history", None)
        except Exception:
            execution_historys = None

        # 把历史注入上下文（并提醒只处理最新请求，别重复旧任务）
        if execution_historys:
            input_context += "\n### Previous Execution Results:\n"
            input_context += json.dumps(execution_historys, indent=2, ensure_ascii=False)
            input_context += "\n\nPlease consider the above execution results when generating the new plan.\n"
            input_context += "Note that the new plan step should only include the user's latest request task and not contain steps from previous tasks."

        response = await self.agent.arun(
            input=input_context,
            stream=False,
            session_id=session_id
        )

        plan_output = response.content
        # LLM 输出可能是 ```json 代码块或纯文字，统一提取成 dict
        plan_output_format = self.extract_plan_from_content(plan_output)

        return plan_output_format

    def inject_execution_results(self, session_id, plan, execution_results: Dict[int, Any]) -> Dict:
        """执行后回写记忆（Task Memory 的写路径）。

        把「本轮计划 + 每步执行结果」追加进 execution_history，存回 SQLite，
        下次 generate_plan 时就能读出来。
        """
        try:
            current_state = self.agent.get_session_state(session_id).get("execution_history", [])
        except Exception:
            current_state = []

        # 追加本轮记录（注意是 append，历史会越攒越长）
        current_state.append(
            {
                "plan": plan,
                "execution_results": execution_results
            }
        )

        self.agent.update_session_state(
            session_state_updates={"execution_history": current_state},
            session_id=session_id
        )


class ActAgent:
    """执行 Agent：按 PlanAgent 的计划逐个调用 MCP 工具。

    与 PlanAgent 的关键差异：**绑定了 tools=[mcp_tools]**，有调用工具的能力。
    tool_call_limit=1 让每步只调一次工具，结果可验证后再进下一步。
    """

    def __init__(self, mcp_tools: MultiMCPTools, act_db=None):
        self.mcp_tools = mcp_tools

        # 执行模型配置（act_model_* 段），通常和规划模型不同
        act_model_provider = config.get('act_model_provider', 'openai')
        act_model_id = config.get('act_model_id', 'gpt-5-2025-08-07')
        act_model_api_key = config.get('act_model_api_key', '')
        act_model_base_url = config.get('act_model_base_url', '')
        act_model_extra_params = config.get('act_model_extra_params', '')

        self.agent = Agent(
            name="Univideo Act Agent",
            model=create_model(
                provider=act_model_provider,
                model_id=act_model_id,
                api_key=act_model_api_key,
                base_url=act_model_base_url or None,
                extra_params=act_model_extra_params,
            ),
            tools=[mcp_tools],  # 有工具：执行 Agent 能真正调 MCP 工具
            instructions="""
            # Intelligent Video Task Execution Assistant

            ## Core Role Definition
            You are an intelligent video task execution assistant (Video Act LLM), specialized in executing video generation, editing, and processing plans. You follow plans provided by the Plan Model while conducting intelligent thinking, calling, wait and feedback throughout the video production workflow.

            ## Output Format
            After received the tool execution result, you should output a JSON object with the following format:
            {
                "success": "True/False",
                "message": "...",
                "content": "...",
                "output_path": "/path/to/generated/file.mp4"
            }
            Important: You MUST fill in the ACTUAL values returned by the tool. Do NOT copy the example text above.
            "output_path" must be the exact file path returned by the tool, not a placeholder description.
            """,
            tool_call_limit=1,
        )

    async def execute_plan(self, question, plan) -> Dict[str, Any]:
        """按计划逐步执行，返回 {步骤号: 该步结果} 字典。

        每步执行后都会 update_plan 把结果写回计划（状态回填）。
        """
        execution_results: Dict[int, Any] = {}

        for idx, step in enumerate(plan['execution_plan']['steps']):
            try:
                logger.info(f"Executing step {idx+1}: {step.get('action_description', 'Unknown')}")
                result = await self._execute_step(step, question, plan, execution_results)

                execution_results[idx+1] = result
                plan = self.update_plan(plan, result, idx)  # 状态回写计划
                logger.info(f"Step {idx+1} completed successfully")

            except Exception as e:
                error_msg = f"step {idx+1} execution failed: {str(e)}"
                logger.error(error_msg)
                return error_msg

        return execution_results

    async def _execute_step(self, step, question, plan, execution_results) -> Any:
        """执行单步：把「问题 + 全计划 + 已完成步骤 + 当前步骤」拼成上下文，
        让 ActAgent 只处理当前这一步（论文：每步只做一个动作）。
        """
        input_context = f"""
        ### User Request
        {question}

        ### Whole Plan
        {plan['execution_plan']}

        ### Completed Steps
        {execution_results}

        ### Current Step
        {step}

        You can only perform the tasks specified in the current step.
        """

        response = await self.agent.arun(
            input=input_context,
            stream=False
        )

        result = self.extract_json(response.content)

        if result is None:
            logger.warning(f"No JSON found in response, treating as text message. Content: {response.content[:200]}...")
            return {
                'success': False,
                'message': response.content,
                'content': response.content,
                'output_path': None
            }

        logger.info(f"Extracted result: {result}")

        self.current_step = {
            "step": step,
            "result": result
        }

        return result

    def extract_json(self, content: str) -> Optional[Dict]:
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                json_str = match.group(1)
                logger.info("Found JSON in code block format")
            else:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1]
                    logger.info("Found JSON without code block")
                else:
                    logger.warning("No JSON structure found in content")
                    return None

            parsed_content = json.loads(json_str)
            logger.info(f"Successfully parsed JSON with keys: {list(parsed_content.keys())}")
            return parsed_content
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed JSON string: {json_str[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in extract_json: {e}")
            return None

    def update_plan(self, plan, step_result, idx):
        """把单步执行结果回填进计划：更新该步 status（成功/失败）和 output。

        执行后计划带着真实结果，PlanAgent 下轮规划就能看到。
        """
        if step_result is None:
            logger.error(f"step_result is None for step {idx+1}")
            plan['execution_plan']['steps'][idx]['status'] = 'failed'
            plan['execution_plan']['steps'][idx]['output'] = 'Step result is None'
            return plan

        if not isinstance(step_result, dict):
            logger.error(f"step_result is not a dict for step {idx+1}: {type(step_result)}")
            plan['execution_plan']['steps'][idx]['status'] = 'failed'
            plan['execution_plan']['steps'][idx]['output'] = f'Invalid result type: {type(step_result)}'
            return plan

        plan['execution_plan']['steps'][idx]['status'] = step_result.get('success', False)
        plan['execution_plan']['steps'][idx]['output'] = step_result.get('output_path') or step_result.get('content', 'No output')

        return plan


# =============================================================================
# SingleAgent：结论2 对照（无 Plan-Act 分工）
# 一个 Agent 同时承担规划与执行：先输出计划 JSON → 自己连环调用工具执行
# → 最后输出带执行状态更新的完整计划 JSON。tool_call_limit=50 是关键，
# 允许一次对话内多次调用工具（Plan-Act 的 ActAgent 每步 limit=1）。
# =============================================================================
SINGLE_ACT_INSTRUCTIONS = """你是视频任务的规划者兼执行者（Single Agent，没有独立的 Plan/Act 分工）。

工作流程（一轮完成全部）：
1. 首先根据用户请求，输出符合下方格式的完整执行计划 JSON（execution_plan，含 steps）；
2. 然后立即自己调用工具，按顺序执行计划中的每个 step；
3. 全部执行完毕后，最后输出【更新后的完整计划 JSON】——每个 step 的 status 填实际执行结果（success 或 failed），output 填实际产出（成功时填工具返回的文件路径）。

必须遵守：
- 每个 step 只做一件事，只调用一个工具；
- 同一个工具最多调用一次，禁止重复生成/重复提交（视频生成很贵）；
- output_path / output 必须填工具真实返回的文件路径，禁止编造；
- 某步失败时如实填 status=failed，并继续尝试剩余步骤（不要整体放弃）；
- 最终输出必须且只能是更新后的计划 JSON，不要附加解释文字。"""


class SingleAgent:
    """结论2 对照组：Single Agent（一个模型同时规划+执行）。

    与 PlanActSystem（PlanAgent 规划 → ActAgent 执行）对比使用：
    相同模型、相同工具集、相同 session 记忆，仅差异在"分工"。
    """

    def __init__(self, mcp_tools: MultiMCPTools = None, db=None,
                 max_run_seconds: int = 600, tool_call_limit: int = 10):
        """Args:
            mcp_tools: MCP 工具。传 None 时构造纯规划 Agent（无工具，零成本，
                用于 plan_only 对比 Planner 差异）。
            max_run_seconds: 单轮总超时（秒），防止模型循环调用烧钱。
            tool_call_limit: 单轮工具调用上限（一次生成约 $0.3-0.5，控制成本）。
        """
        self.mcp_tools = mcp_tools
        self.max_run_seconds = max_run_seconds

        plan_prompt = load_prompt("plan")
        full_instructions = f"{plan_prompt}\n\n{SINGLE_ACT_INSTRUCTIONS}"

        model_provider = config.get('plan_model_provider', 'openai')
        model_id = config.get('plan_model_id', 'gpt-5-2025-08-07')
        model_api_key = config.get('plan_model_api_key', '')
        model_base_url = config.get('plan_model_base_url', '')
        model_extra_params = config.get('plan_model_extra_params', '')

        self.agent = Agent(
            name="Univideo Single Agent",
            model=create_model(
                provider=model_provider,
                model_id=model_id,
                api_key=model_api_key,
                base_url=model_base_url or None,
                extra_params=model_extra_params,
            ),
            instructions=full_instructions,
            tools=[mcp_tools] if mcp_tools else [],
            tool_call_limit=tool_call_limit,  # 关键：一次对话内规划+连环调工具（上限控制成本）
            db=db,
            add_history_to_context=True,
            num_history_messages=10,
            session_state={"execution_history": []},
        )

    async def run(self, session_id, user_request: str) -> Dict[str, Any]:
        """单轮执行：规划 + 执行 + 返回更新后的计划。

        Returns:
            与 PlanActSystem.execute_task 相同的 {plan, execution} 结构，
            其中 execution[step_num] = {success, output_path, message}，
            便于 runner 用同一套解析逻辑。
        """
        # 成本护栏：单轮总超时，防止模型循环调用工具/重复生成
        response = await asyncio.wait_for(
            self.agent.arun(
                input=user_request,
                stream=False,
                session_id=session_id,
            ),
            timeout=self.max_run_seconds,
        )

        plan = extract_plan_from_content(response.content)

        # 防御：没解析出计划 JSON
        if not isinstance(plan, dict) or 'execution_plan' not in plan:
            return {
                "plan": plan,
                "execution": {
                    1: {
                        "success": False,
                        "message": f"SingleAgent 未能生成有效计划: {str(plan)[:200]}",
                        "output_path": None,
                    }
                }
            }

        # 把更新后的计划 steps 转成 execution 字典（与 PlanAct 输出对齐）
        execution: Dict[int, Any] = {}
        steps = plan.get('execution_plan', {}).get('steps', [])
        for i, step in enumerate(steps):
            status = step.get('status', 'failed')
            output = step.get('output') or step.get('output_path')
            execution[i + 1] = {
                'success': str(status).lower() in ('true', 'success', 'completed'),
                'output_path': output if (output and os.path.exists(str(output))) else None,
                'message': f"{step.get('action_description', '')} -> {step.get('tool', {})}",
            }

        return {"plan": plan, "execution": execution}


class PlanActSystem:
    """总装配：连接 MCP 工具、持有记忆数据库、编排三个 Agent。

    对外两个入口：
      - execute_task()     ：Plan-Act 双 Agent 主流程
      - execute_single()   ：Single-Agent 对照（结论2 实验）
    """

    def __init__(self, mcp_command: List[str], db_file: str = "plan_act_system"):
        mcp_tools_path = config.get('mcp_tools_path')
        # MultiMCPTools：管理多个 MCP server 的连接（每个 server 是独立子进程）
        self.mcp_tools = MultiMCPTools(
            commands=mcp_command,
            env={
                "PYTHONPATH": mcp_tools_path,
                "CWD": mcp_tools_path
            },
            timeout_seconds=600,
            refresh_connection=True
        )
        # Task 记忆数据库：plan_act_system_plan.db（SQLite，表 agno_sessions）
        self._plan_db = SqliteDb(db_file=f"{db_file}_plan.db")
        self.plan_agent = None
        self.act_agent = None
        self.single_agent = None

    async def __aenter__(self):
        """async 上下文进入：先连接 MCP，再创建两个 Agent（懒加载到 execute 时才用）。"""
        await self.mcp_tools.connect()
        self.plan_agent = PlanAgent(self.mcp_tools, self._plan_db)
        self.act_agent = ActAgent(self.mcp_tools)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """async 上下文退出：关闭所有 MCP 子进程。"""
        await self.mcp_tools.close()

    async def execute_single(self, session_id, user_request: str) -> Dict[str, Any]:
        """Single-Agent 对照执行：同一系统同一 MCP 连接，只换执行方式。

        结论2 实验用：Plan-Act vs Single-Agent 的差异仅在于分工方式，
        MCP 工具、模型、session 记忆都保持一致。
        """
        if self.single_agent is None:
            self.single_agent = SingleAgent(self.mcp_tools, self._plan_db)
        return await self.single_agent.run(session_id, user_request)

    async def execute_task(self, session_id, user_request: str) -> Dict[str, Any]:
        """Plan-Act 主流程（论文核心）：规划 → 执行 → 记忆回写，共三步。"""
        # ① 规划：PlanAgent 拆出执行计划（JSON）
        execution_plan = await self.plan_agent.generate_plan(session_id, user_request)

        # 防御：PlanAgent 解析失败时返回的是字符串而非 dict
        if not isinstance(execution_plan, dict) or 'execution_plan' not in execution_plan:
            return {
                "plan": execution_plan,
                "execution": {
                    1: {
                        "success": False,
                        "message": f"PlanAgent 未能生成有效计划: {str(execution_plan)[:200]}",
                        "output_path": None,
                    }
                }
            }

        # ② 执行：ActAgent 逐步骤调用 MCP 工具，返回 {步骤号: 结果}
        results = await self.act_agent.execute_plan(user_request, execution_plan)
        # ③ 记忆回写：把「计划 + 结果」存进 SQLite（Task Memory 写路径）
        self.plan_agent.inject_execution_results(session_id, execution_plan, results)

        return {
            "plan": execution_plan,
            "execution": results
        }

    async def execute_task_stream(self, session_id, user_request: str):
        try:
            logger.info(f"[Stream] Starting task stream for session {session_id}")
            yield {'type': 'content', 'content': 'start to generate plans...'}

            execution_plan = await self.plan_agent.generate_plan(session_id, user_request)
            if not isinstance(execution_plan, dict) or 'execution_plan' not in execution_plan:
                logger.error(f"[Stream] Plan generation failed for session {session_id}")
                yield {'type': 'content', 'content': f'no valid plan generated.\n{execution_plan}'}
                return

            logger.info(f"[Stream] Plan generated successfully with {len(execution_plan.get('execution_plan', {}).get('steps', []))} steps")
            todo_events = generate_todo_progress_event(execution_plan)
            logger.info(f"[Stream] Yielding todo_progress event")
            yield todo_events

            execution_results: Dict[int, Any] = {}
            steps = execution_plan.get('execution_plan', {}).get('steps', [])

            for idx, step in enumerate(steps):
                step_num = idx + 1
                logger.info(f"[Stream] Starting execution of step {step_num}/{len(steps)}: {step}")
                try:
                    action_desc = step.get('action_description', f'step {step_num}')
                    tool_name = step.get('tool', {}).get('name', 'unknown')

                    tool_start_event = {'type': 'tool_start', 'tool': tool_name}
                    yield tool_start_event

                    logger.info(f"[Stream] Executing step {step_num}: {action_desc}")
                    result = await self.act_agent._execute_step(
                        step, user_request, execution_plan, execution_results
                    )

                    execution_results[step_num] = result
                    execution_plan = self.act_agent.update_plan(execution_plan, result, idx)

                    tool_end_event = {'type': 'tool_end', 'result': result}
                    yield tool_end_event

                    if result and isinstance(result, dict):
                        success = result.get('success')
                        is_success = success in [True, 'True', 'true']

                        if not is_success:
                            error_msg = result.get('message', 'Step execution failed')
                            logger.error(f"[Stream] Step {step_num} failed: {error_msg}")
                            yield {
                                'type': 'error',
                                'content': f"Task failed at step {step_num}: {error_msg}"
                            }
                            return

                    logger.info(f"[Stream] Step {step_num} completed successfully")

                except Exception as e:
                    error_msg = f"step {idx+1} execution failed: {str(e)}"
                    logger.error(f"[Stream] {error_msg}")
                    logger.error(traceback.format_exc())
                    yield {'type': 'error', 'content': f"Error executing step {idx+1}\n {error_msg}"}
                    return

            logger.info(f"[Stream] Injecting execution results back to plan agent")
            self.plan_agent.inject_execution_results(session_id, execution_plan, execution_results)

            finish_event = {'type': 'finish', 'session_id': session_id}
            yield finish_event

            logger.info(f"[Stream] Task stream completed successfully for session {session_id}")

        except Exception as e:
            logger.error(f"[Stream] Error in execute_task_stream: {e}")
            logger.error(traceback.format_exc())
            yield {'type': 'error', 'content': str(e)}


async def initialize_global_agents(mcp_config_path: str = None) -> PlanActSystem:
    """工厂函数：读 MCP 配置 → 启动所有 server 子进程 → 返回就绪的 PlanActSystem。

    mcp_config_path 传 eval/mcp_configs_eval.json 可指定评测用配置（哪些 server 启动）。
    """
    config_path = mcp_config_path or config.get('mcp_servers_config')

    try:
        # 从 JSON 里读出每个 server 的启动命令（如 "python -m mcp_tools.video_gen"）
        with open(config_path, 'r', encoding='utf-8') as f:
            mcp_config = json.load(f)

        mcp_servers = mcp_config.get("mcpServers", {})
        logger.info(f"Loaded {len(mcp_servers)} MCP servers from config")

        mcp_commands = []
        for server_name, server_config in mcp_servers.items():
            command = server_config.get("command", "")
            args = server_config.get("args", [])
            full_command = f"{command} {' '.join(args)}"
            mcp_commands.append(full_command)
            logger.info(f"Registered MCP server '{server_name}': {full_command}")

    except FileNotFoundError:
        logger.warning(f"MCP config file not found: {config_path}, using default")
        mcp_commands = ["npx -y @modelcontextprotocol/server-filesystem /tmp"]
    except Exception as e:
        logger.error(f"Error loading MCP config: {e}, using default")
        mcp_commands = ["npx -y @modelcontextprotocol/server-filesystem /tmp"]

    # 启动并连接所有 MCP 子进程
    global_plan_act_system = PlanActSystem(mcp_command=mcp_commands)
    await global_plan_act_system.__aenter__()

    logger.info("Global PlanActSystem initialized")
    return global_plan_act_system


def format_result(result: Dict) -> str:
    """把 execute_task 返回的 {plan, execution} 格式化成缩进的完整 JSON。

    保留原始层级结构与完整信息（不截断 message/content），
    用 2 空格缩进让嵌套关系清晰可读。
    """
    if not result:
        return "(空结果)"
    return json.dumps(result, indent=2, ensure_ascii=False)


async def main():
    """交互式 CLI 入口：python univa_agent.py 后输入请求即可执行 Plan-Act 链路。

    所有交互共用同一个 session_id → 共享 Task 记忆（能看到跨轮记忆效果）。
    """
    system = await initialize_global_agents()
    session_id = "test_interactive_session_001"

    try:
        print("system is ready. You can start inputting tasks (type 'exit' or 'quit' to stop).")
        while True:
            try:
                input_prompt = input("\nUser (please input propmt): ")
                if input_prompt.lower() in ['exit', 'quit']:
                    break

                if not input_prompt.strip():
                    continue

                print(f"processing: {input_prompt} ...")

                result = await system.execute_task(session_id, input_prompt)

                print("\n" + "="*60)
                print(format_result(result))
                print("\ncompleted.")
                print("="*60)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"error: {e}")
    finally:
        await system.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
