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
    def __init__(self, mcp_tools: MultiMCPTools, plan_db):
        self.mcp_tools = mcp_tools

        plan_prompt = load_prompt("plan")
        full_instructions = f"{plan_prompt}\n"

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
            instructions=full_instructions,
            db=plan_db,
            add_history_to_context=True,
            num_history_messages=10,
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
        input_context = f"User Request: {user_request}\n"

        try:
            execution_historys = self.agent.get_session_state(session_id).get("execution_history", None)
        except Exception:
            execution_historys = None

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
        plan_output_format = self.extract_plan_from_content(plan_output)

        return plan_output_format

    def inject_execution_results(self, session_id, plan, execution_results: Dict[int, Any]) -> Dict:
        try:
            current_state = self.agent.get_session_state(session_id).get("execution_history", [])
        except Exception:
            current_state = []

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
    def __init__(self, mcp_tools: MultiMCPTools, act_db=None):
        self.mcp_tools = mcp_tools

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
            tools=[mcp_tools],
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
        execution_results: Dict[int, Any] = {}

        for idx, step in enumerate(plan['execution_plan']['steps']):
            try:
                logger.info(f"Executing step {idx+1}: {step.get('action_description', 'Unknown')}")
                result = await self._execute_step(step, question, plan, execution_results)

                execution_results[idx+1] = result
                plan = self.update_plan(plan, result, idx)
                logger.info(f"Step {idx+1} completed successfully")

            except Exception as e:
                error_msg = f"step {idx+1} execution failed: {str(e)}"
                logger.error(error_msg)
                return error_msg

        return execution_results

    async def _execute_step(self, step, question, plan, execution_results) -> Any:
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

    def __init__(self, mcp_tools: MultiMCPTools, db=None,
                 max_run_seconds: int = 600, tool_call_limit: int = 10):
        """Args:
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
            tools=[mcp_tools],
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
    def __init__(self, mcp_command: List[str], db_file: str = "plan_act_system"):
        mcp_tools_path = config.get('mcp_tools_path')
        self.mcp_tools = MultiMCPTools(
            commands=mcp_command,
            env={
                "PYTHONPATH": mcp_tools_path,
                "CWD": mcp_tools_path
            },
            timeout_seconds=600,
            refresh_connection=True
        )
        self._plan_db = SqliteDb(db_file=f"{db_file}_plan.db")
        self.plan_agent = None
        self.act_agent = None
        self.single_agent = None

    async def __aenter__(self):
        await self.mcp_tools.connect()
        self.plan_agent = PlanAgent(self.mcp_tools, self._plan_db)
        self.act_agent = ActAgent(self.mcp_tools)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

        results = await self.act_agent.execute_plan(user_request, execution_plan)
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
    config_path = mcp_config_path or config.get('mcp_servers_config')

    try:
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
