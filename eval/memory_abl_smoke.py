"""
三层记忆消融 · 无 MCP 冒烟（零成本）
====================================
不带 MCP 工具、不调生成 API，只验证 Plan/Single 两个 Agent 在
不同 memory_cfg 下规划的注入是否生效、计划是否有效。

5 种 memory_cfg（结论3 消融）：
  none / task / task+user / task+global / task+user+global

用法：
  python -m eval.memory_abl_smoke --tasks lt2v --items 0 --run-id abl_smoke
输出：
  eval/outputs/<run_id>/summary.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

BENCH_ROOT = Path("D:/univa/datasets/UniVA-Bench")
OUT_ROOT = Path("D:/univa/eval/outputs")
MEMORY_CFGS = ["none", "task", "task+user", "task+global", "task+user+global"]


def build_request(task: str, item: dict) -> str:
    if task == "lt2v":
        return item["prompt"]
    if task == "it2v":
        ref = str(BENCH_ROOT / item["sources"][0])
        return f"用 image2video_gen 工具，基于参考图 {ref} 生成 5 秒视频。描述: {item['prompt'][:200]}"
    return str(item.get("prompt", ""))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="lt2v")
    parser.add_argument("--items", default="0")
    parser.add_argument("--run-id", default="abl_smoke")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "univa"))
    from univa.univa_agent import PlanAgent, SingleAgent
    from univa.memory.user_memory import UserMemory
    from univa.memory.global_memory import GlobalMemory

    # 记忆实例（无 MCP 也能用）
    user_mem = UserMemory(db_path=str(Path(__file__).resolve().parents[1] / "univa" / "memory_user.db"))
    user_mem.seed_default("default_user")
    global_mem = GlobalMemory()

    # 取任务 prompt
    prompts = []
    for task in args.tasks.split(","):
        lines = [json.loads(l) for l in open(BENCH_ROOT / f"{task}.jsonl", encoding="utf-8")]
        for idx in map(int, args.items.split(",")):
            if idx < len(lines):
                prompts.append((task, lines[idx]))

    run_dir = OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for mem in MEMORY_CFGS:
        um = user_mem if "user" in mem else None
        gm = global_mem if "global" in mem else None
        pa = PlanAgent(None, None, user_memory=um, global_memory=gm)
        sa = SingleAgent(None, None, user_memory=um, global_memory=gm)

        for ti, (task, item) in enumerate(prompts):
            item_id = item.get("id") or item.get("videoID")
            prompt = build_request(task, item)

            # Plan-Act 规划（无工具）
            p_plan = await pa.generate_plan(f"{args.run_id}|{mem}|{ti}", prompt, memory_cfg=mem)
            # Single 规划（无工具）
            s_res = await sa.run(f"{args.run_id}|{mem}|{ti}_s", prompt, memory_cfg=mem)
            s_plan = s_res.get("plan")

            p_ok = isinstance(p_plan, dict) and "execution_plan" in p_plan and len(p_plan.get("execution_plan", {}).get("steps", [])) > 0
            s_ok = isinstance(s_plan, dict) and "execution_plan" in s_plan and len(s_plan.get("execution_plan", {}).get("steps", [])) > 0

            def tools_of(plan):
                if not isinstance(plan, dict):
                    return []
                return [str(s.get("tool", {}).get("name", ""))[:30] for s in plan.get("execution_plan", {}).get("steps", [])]

            row = {
                "memory_cfg": mem, "task": task, "item_id": item_id,
                "planact": {"valid": p_ok, "steps": len(p_plan.get("execution_plan", {}).get("steps", [])) if p_ok else 0,
                            "tools": tools_of(p_plan)},
                "single": {"valid": s_ok, "steps": len(s_plan.get("execution_plan", {}).get("steps", [])) if s_ok else 0,
                           "tools": tools_of(s_plan)},
            }
            all_rows.append(row)
            print(f"[{mem:18}] {item_id}: Plan-Act valid={p_ok} tools={tools_of(p_plan)} | "
                  f"Single valid={s_ok} tools={tools_of(s_plan)}")

    # 汇总
    summary = {"run_id": args.run_id, "rows": all_rows}
    for role in ["planact", "single"]:
        summary[f"{role}_valid"] = sum(1 for r in all_rows if r[role]["valid"])
        summary[f"{role}_total"] = len(all_rows)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n计划有效: Plan-Act={summary['planact_valid']}/{summary['planact_total']}  "
          f"Single={summary['single_valid']}/{summary['single_total']}")
    print(f"结果: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
