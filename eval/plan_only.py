"""
Planner 差异对比（论文 5.2 · 只规划不执行，零生成成本）
=======================================================
对同一批 prompt，分别看两种架构的 Planner 会输出什么计划：

  - Plan-Act  : PlanAgent.generate_plan()   （双 Agent 的规划 Agent，无工具）
  - Single    : SingleAgent(None) 纯规划版   （单 Agent 架构，无工具，只输出计划）

不执行 MCP 工具、不调生成 API——纯 LLM 规划，一次调用约几分钱。

用法：
  python -m eval.plan_only --tasks lt2v,it2v --items 0,1,2,3,4 \
    --mcp-config eval/mcp_configs_eval.json --run-id plan_lt2v

输出：
  eval/outputs/<run_id>/<task>/<item_id>.json   逐条双计划对比
  eval/outputs/<run_id>/summary.json            汇总（计划有效性/步骤数/工具选择）
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

BENCH_ROOT = Path("D:/univa/datasets/UniVA-Bench")
OUT_ROOT = Path("D:/univa/eval/outputs")

TASK_FILES = {
    "lt2v": "lt2v.jsonl", "it2v": "it2v.jsonl", "v2v": "v2v.jsonl",
    "lve": "lve.jsonl", "lvu": "lvu.jsonl", "seg": "seg.jsonl",
}
EXPECTED_TOOLS = {
    "lt2v": "text2video_gen", "it2v": "image2video_gen", "v2v": "style_transfer",
    "lve": "long_video_edit", "lvu": "vision2text_gen", "seg": "video_referring_segmentation",
}


def build_request(task: str, item: dict) -> str:
    """复用 runner 的请求构造，保证两种模式输入完全一致。"""
    if task == "lt2v":
        return item["prompt"]
    if task == "it2v":
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
        return (f"请分析视频 {video} 回答选择题，用 vision2text_gen。问题: {q['question']} "
                f"选项: {' '.join(q['options'])}。给出答案字母")
    if task == "seg":
        src = str(BENCH_ROOT / item["sources"][0])
        return (f"分割视频 {src} 中的目标: {item['prompt']}")
    raise ValueError(f"未知任务: {task}")


def summarize_plan(plan, task: str):
    """从计划里提取结构化信息（有效性 / 步骤数 / 工具选择）。"""
    if not isinstance(plan, dict) or "execution_plan" not in plan:
        return {
            "valid": False,
            "steps": 0,
            "tools": [],
            "desc": [],
            "raw_head": str(plan)[:200],
        }
    steps = plan["execution_plan"].get("steps", [])
    tools = []
    for s in steps:
        t = s.get("tool", {})
        if isinstance(t, dict):
            tools.append(t.get("name") or t.get("type") or str(t)[:40])
        else:
            tools.append(str(t)[:40])
    return {
        "valid": len(steps) > 0,
        "steps": len(steps),
        "tools": tools,
        "desc": [s.get("action_description", "")[:80] for s in steps],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--items", default="0,1,2")
    parser.add_argument("--mcp-config", default=None)
    parser.add_argument("--run-id", default="plan_only")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "univa"))
    from univa.univa_agent import initialize_global_agents, SingleAgent

    tasks = [t.strip() for t in args.tasks.split(",")]
    items = [int(i) for i in args.items.split(",")]

    run_dir = OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    system = await initialize_global_agents(mcp_config_path=args.mcp_config)
    # SingleAgent(None)：无工具纯规划版，零成本
    single = SingleAgent(None)

    all_rows = []
    for task in tasks:
        jsonl = BENCH_ROOT / TASK_FILES[task]
        if not jsonl.exists():
            print(f"[跳过] {jsonl}")
            continue
        lines = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
        print(f"\n=== {task} ({len(lines)} 条, 测 {items}) ===")
        task_dir = run_dir / task
        task_dir.mkdir(exist_ok=True)

        for idx in items:
            if idx >= len(lines):
                continue
            item = lines[idx]
            item_id = item.get("id") or item.get("videoID")
            prompt = build_request(task, item)
            print(f"  规划 {item_id} ...")

            # 双 Agent：PlanAgent（规划专用）
            dual_plan = await system.plan_agent.generate_plan(
                f"{args.run_id}|dual|{task}|{idx}", prompt)
            # Single：无工具纯规划
            single_result = await single.run(f"{args.run_id}|single|{task}|{idx}", prompt)
            single_plan = single_result.get("plan")

            du = summarize_plan(dual_plan, task)
            su = summarize_plan(single_plan, task)

            row = {
                "task": task,
                "item_id": item_id,
                "expected_tool": EXPECTED_TOOLS.get(task),
                "dual": {"plan_valid": du["valid"], "steps": du["steps"],
                         "tools": du["tools"], "desc": du["desc"],
                         "plan": dual_plan},
                "single": {"plan_valid": su["valid"], "steps": su["steps"],
                           "tools": su["tools"], "desc": su["desc"],
                           "plan": single_plan},
            }
            out_file = task_dir / f"{item_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2, default=str)
            all_rows.append(row)

            print(f"    dual:   valid={du['valid']} steps={du['steps']} tools={du['tools']}")
            print(f"    single: valid={su['valid']} steps={su['steps']} tools={su['tools']}")

    await system.__aexit__(None, None, None)

    # 汇总
    summary = {
        "run_id": args.run_id,
        "total": len(all_rows),
        "dual_plan_valid": sum(1 for r in all_rows if r["dual"]["plan_valid"]),
        "single_plan_valid": sum(1 for r in all_rows if r["single"]["plan_valid"]),
        "dual_expected_tool": sum(1 for r in all_rows
                                  if r["dual"]["plan_valid"] and r["expected_tool"] in r["dual"]["tools"]),
        "single_expected_tool": sum(1 for r in all_rows
                                    if r["single"]["plan_valid"] and r["expected_tool"] in r["single"]["tools"]),
        "rows": [{k: v for k, v in r.items() if k != "plan"} for r in all_rows],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n=== 汇总 ===")
    print(f"计划有效性:  Plan-Act={summary['dual_plan_valid']}/{summary['total']}  "
          f"Single={summary['single_plan_valid']}/{summary['total']}")
    print(f"工具选择正确: Plan-Act={summary['dual_expected_tool']}/{summary['total']}  "
          f"Single={summary['single_expected_tool']}/{summary['total']}")
    print(f"结果: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
