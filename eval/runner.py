"""
UniVA-Bench 实验执行器
======================
读取数据集 jsonl，对每个任务调用 UniVA 系统执行，收集结果并计算指标。

用法：
  python -m eval.runner --tasks lt2v,it2v --items 0,1 --mode agent   # Agent 链路
  python -m eval.runner --tasks lt2v --items 0 --mode direct          # 直连 API
  python -m eval.runner --tasks lvu --items 0 --mode agent            # 理解类

输出：
  eval/outputs/<task>/<id>/manifest.json   每条任务的结果
  eval/outputs/summary.json                汇总（成功后）
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

BENCH_ROOT = Path("D:/univa/datasets/UniVA-Bench")
OUT_ROOT = Path("D:/univa/eval/outputs")

# 各任务类型对应的 jsonl
TASK_FILES = {
    "lt2v": "lt2v.jsonl",   # 长文本→视频
    "it2v": "it2v.jsonl",   # 图片→视频
    "v2v": "v2v.jsonl",     # 视频→视频
    "lve": "lve.jsonl",     # 长视频编辑
    "lvu": "lvu.jsonl",     # 长视频理解
    "seg": "seg.jsonl",     # 长视频分割（需 GPU）
}

# 各任务需要的工具（Agent 链路下 PlanAgent 应选这些）
EXPECTED_TOOLS = {
    "lt2v": "text2video_gen",
    "it2v": "image2video_gen",
    "v2v": "style_transfer",
    "lve": "style_transfer",
    "lvu": "vision2text_gen",
    "seg": "video_referring_segmentation",
}


def build_request(task: str, item: dict) -> str:
    """按任务类型把 jsonl 条目转成用户请求文本。"""
    if task == "lt2v":
        return item["prompt"]
    if task == "it2v":
        # 用 image2video_gen（单图直连）更稳；entity2video 故事板会幻觉不存在路径
        ref = str(BENCH_ROOT / item["sources"][0])
        return (f"用 image2video_gen 工具，基于参考图 {ref} 生成 5 秒视频。描述: {item['prompt'][:300]}")
    if task == "v2v":
        src = str(BENCH_ROOT / item["sources"][0])
        return (f"对源视频 {src} 做视频到视频转换: {item['prompt'][:200]}")
    if task == "lve":
        src = str(BENCH_ROOT / item["source"][0])
        return (f"编辑视频 {src}: {item['prompt']}")
    if task == "lvu":
        video = str(BENCH_ROOT / f"Videos/{item['videoID']}.mp4")
        q = item["qas"][0]
        opts = " ".join(q["options"])
        return (f"请分析视频 {video} 来回答下面的选择题，用 vision2text_gen 工具理解视频内容。"
                f"问题: {q['question']} 选项: {opts}。最终给出答案字母")
    if task == "seg":
        src = str(BENCH_ROOT / item["sources"][0])
        return (f"分割视频 {src} 中的目标: {item['prompt']}")
    raise ValueError(f"未知任务: {task}")


def run_direct(task: str, item: dict, api_key: str) -> dict:
    """调试用：直连 API 执行（不走 Agent 链路）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "univa"))
    from utils import wavespeed_api as wa

    if task == "lt2v":
        return wa.text_to_video_generate(api_key, item["prompt"], save_path=None)
    if task == "it2v":
        img = str(BENCH_ROOT / item["sources"][0])
        return wa.image_to_video_generate(api_key, item["prompt"][:200], img)
    if task in ("v2v", "lve"):
        src = item.get("source") or item.get("sources") or [""]
        src = str(BENCH_ROOT / (src[0] if not src[0].startswith("D:") else src[0]))
        return {"success": False, "message": "direct 模式需分片，请用 agent 模式"}
    if task == "lvu":
        from utils.query_llm import multimodal_query, query_openai, llm_config
        video = str(BENCH_ROOT / f"Videos/{item['videoID']}.mp4")
        qa_results = []
        for q in item["qas"]:
            opts = q.get("options") or []
            is_mcq = bool(opts)
            if is_mcq:
                prompt = (f"Answer the question based on the video. Question: {q['question']} "
                          f"Options: {' '.join(opts)} Reply with just the letter.")
            else:
                prompt = (f"Answer the question based on the video. "
                          f"Question: {q['question']}")
            answer = multimodal_query(prompt, video_path=video, video_frames_to_extract=64)
            correct = q.get("answer", "")

            if is_mcq:
                # 选择题：字母精确匹配
                ok = bool(answer and str(answer).strip().upper() == str(correct).upper())
            else:
                # 简答题：纯文本 LLM judge（不带图，判断是否抓住核心点）
                judge_prompt = (
                    f"判断下面的模型回答是否准确抓住了参考回答的核心要点。"
                    f"只需输出 YES 或 NO。\n\n"
                    f"问题: {q['question']}\n"
                    f"参考回答: {correct[:600]}\n"
                    f"模型回答: {str(answer)[:600]}\n"
                )
                judge_resp = query_openai(
                    api_key=llm_config.get('openai_api_key', None),
                    model=llm_config.get('model', 'gpt-5-2025-08-07'),
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_completion_tokens=32,
                    base_url=llm_config.get('base_url', 'https://api.openai.com/v1')
                )
                ok = "YES" in str(judge_resp.get("content", "")).upper()

            qa_results.append({
                "question": q["question"][:120],
                "type": "mcq" if is_mcq else "open",
                "options": opts,
                "answer": str(answer)[:50],
                "correct": correct[:120],
                "ok": ok,
            })
        n_ok = sum(1 for r in qa_results if r["ok"])
        accuracy = n_ok / len(qa_results)
        return {
            "success": accuracy >= 0.5,  # 跑完即算完成，>50% 算达标
            "accuracy": round(accuracy, 4),
            "n_ok": n_ok,
            "n_total": len(qa_results),
            "qa_results": qa_results,
            "output_path": None,
        }
    return {"success": False, "message": f"direct 不支持 {task}"}


async def run_agent_item(system, task: str, item: dict, session_id: str,
                         method: str = "planact") -> dict:
    """走 Agent 链路执行单个任务。

    method: "planact"=Plan-Act 双 Agent（默认）；"single"=Single Agent 对照。
    """
    request = build_request(task, item)
    try:
        if method == "single":
            result = await system.execute_single(session_id, request)
        else:
            result = await system.execute_task(session_id, request)
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()[-800:]}

    # 解析执行结果（execution 可能是 dict 或错误字符串）
    execution = result.get("execution", {})
    step_results = []
    if isinstance(execution, dict):
        for step_num, step_res in execution.items():
            if isinstance(step_res, dict):
                step_results.append({
                    "step": step_num,
                    "success": step_res.get("success"),
                    "output_path": step_res.get("output_path"),
                    "message": str(step_res.get("message", ""))[:200],
                })
            else:
                step_results.append({
                    "step": step_num,
                    "success": False,
                    "output_path": None,
                    "message": f"步骤返回非字典: {str(step_res)[:100]}",
                })
    elif isinstance(execution, str):
        step_results.append({
            "step": 1, "success": False, "output_path": None,
            "message": f"执行返回错误: {execution[:200]}",
        })

    # 汇总：生成/编辑类看 output_path，理解类（lvu）看 message 内容
    is_lvu = task == "lvu"
    any_success = any(
        s.get("success") in [True, "True", "true"]
        and (s.get("output_path") if not is_lvu else s.get("message"))
        for s in step_results
    )
    output_path = next(
        (s["output_path"] for s in step_results if s.get("output_path")), None
    )

    return {
        "task": task,
        "item_id": item.get("id") or item.get("videoID"),
        "request": request[:200],
        "success": any_success,
        "output_path": output_path,
        "steps": step_results,
        "raw": {k: str(v)[:300] for k, v in result.items() if k != "plan"} if isinstance(result, dict) else str(result)[:300],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="逗号分隔: lt2v,it2v,v2v,lve,lvu,seg")
    parser.add_argument("--items", default="0", help="条目索引，逗号分隔: 0,1,2")
    parser.add_argument("--mode", default="agent", choices=["agent", "single", "direct"])
    parser.add_argument("--memory", default=None,
                        help="记忆消融: none/task/task+user/task+global/task+user+global")
    parser.add_argument("--mcp-config", default=None, help="MCP 配置文件路径")
    parser.add_argument("--run-id", default="run1")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    items = [int(i) for i in args.items.split(",")]

    # 准备输出目录
    run_dir = OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("agent", "single"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "univa"))
        from univa.univa_agent import initialize_global_agents
        system = await initialize_global_agents(mcp_config_path=args.mcp_config,
                                                memory_cfg=args.memory)
        method = "single" if args.mode == "single" else "planact"
    else:
        import os
        from dotenv import load_dotenv
        load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))
        api_key = os.environ["WAVESPEED_API_KEY"]

    all_results = []
    for task in tasks:
        if task not in TASK_FILES:
            print(f"[跳过] 未知任务 {task}")
            continue
        jsonl_path = BENCH_ROOT / TASK_FILES[task]
        if not jsonl_path.exists():
            print(f"[跳过] {jsonl_path} 不存在")
            continue

        lines = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        print(f"\n=== {task} ({len(lines)} 条, 测 {items}) ===")
        for idx in items:
            if idx >= len(lines):
                print(f"  [跳过] 索引 {idx} 超出范围")
                continue
            item = lines[idx]
            item_id = item.get("id") or item.get("videoID")
            task_dir = run_dir / task
            task_dir.mkdir(exist_ok=True)

            print(f"  执行 {item_id} ...")
            # 即时标记：进程中断也能知道这条跑到哪一步
            pending_file = task_dir / f"{item_id}.pending.json"
            with open(pending_file, "w", encoding="utf-8") as f:
                json.dump({"item_id": item_id, "task": task, "started": True,
                           "request": build_request(task, item)[:200]}, f,
                          ensure_ascii=False, indent=2)

            if args.mode in ("agent", "single"):
                result = await run_agent_item(system, task, item, f"{args.run_id}|{task}|{idx}",
                                              method=method)
            else:
                result = run_direct(task, item, api_key)
                result["task"] = task
                result["item_id"] = item_id

            # 保存 manifest（完成后删除 pending 标记）
            manifest = {**result, "item": {k: str(v)[:500] for k, v in item.items() if k != "qas"}}
            out_file = task_dir / f"{item_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            if pending_file.exists():
                pending_file.unlink()

            status = "✅" if result.get("success") else "❌"
            print(f"  {status} {item_id}: output={result.get('output_path')}")
            all_results.append(manifest)

    if args.mode in ("agent", "single"):
        await system.__aexit__(None, None, None)

    # 汇总
    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "total": len(all_results),
        "success": sum(1 for r in all_results if r.get("success")),
        "results": all_results,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {summary['success']}/{summary['total']} 成功")
    print(f"结果: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
