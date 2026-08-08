"""
记忆注入效果验证（无 MCP，零成本）
==================================
同一任务，对比 memory_cfg='none' vs 'task+user+global' 下 Plan-Act 的计划，
看记忆注入是否真正改变规划（工具选择 / 步骤描述 / 是否体现领域知识）。

用 lve（长视频编辑）任务——Global 知识库里"长视频编辑必须分片"最相关，
若注入生效，模型计划里应体现"分片/逐段"等知识。

用法：python -m eval.memory_effect --tasks lve --items 0,1 --run-id mem_effect
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

BENCH_ROOT = Path("D:/univa/datasets/UniVA-Bench")
OUT_ROOT = Path("D:/univa/eval/outputs")
MEMORY_CFGS = ["none", "task+user+global"]


def build_request_lve(item: dict) -> str:
    src = item["source"][0] if isinstance(item["source"], list) else item["source"]
    return f"编辑视频 {BENCH_ROOT / src}: {item['prompt']}"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="lve")
    parser.add_argument("--items", default="0,1")
    parser.add_argument("--run-id", default="mem_effect")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "univa"))
    from univa.univa_agent import PlanAgent
    from univa.memory.user_memory import UserMemory
    from univa.memory.global_memory import GlobalMemory

    user_mem = UserMemory(db_path=str(Path(__file__).resolve().parents[1] / "univa" / "memory_user.db"))
    user_mem.seed_default("default_user")
    global_mem = GlobalMemory()

    run_dir = OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks.split(","):
        lines = [json.loads(l) for l in open(BENCH_ROOT / f"{task}.jsonl", encoding="utf-8")]
        for idx in map(int, args.items.split(",")):
            if idx >= len(lines):
                continue
            item = lines[idx]
            item_id = item.get("id") or item.get("videoID")
            prompt = build_request_lve(item)
            print(f"\n========== {task}/{item_id} ==========")
            print(f"请求: {prompt[:100]}...")

            # Global 会检索到什么（确认注入内容相关）
            print(f"\n[Global 检索到的知识]")
            print(global_mem.retrieve(prompt, top_k=2)[:200], "...")

            rows = {}
            for mem in MEMORY_CFGS:
                um = user_mem if "user" in mem else None
                gm = global_mem if "global" in mem else None
                pa = PlanAgent(None, None, user_memory=um, global_memory=gm)
                plan = await pa.generate_plan(f"{args.run_id}|{mem}|{item_id}", prompt, memory_cfg=mem)

                valid = isinstance(plan, dict) and "execution_plan" in plan
                steps = plan.get("execution_plan", {}).get("steps", []) if valid else []
                desc = [s.get("action_description", "")[:80] for s in steps]
                tools = [str(s.get("tool", {}).get("name", "")) for s in steps]

                rows[mem] = {"plan": plan, "valid": valid, "steps": len(steps),
                             "tools": tools, "desc": desc}
                print(f"\n[{mem}] valid={valid} steps={len(steps)} tools={tools}")
                for d in desc:
                    print(f"    · {d}")

            with open(run_dir / f"{task}_{item_id}.json", "w", encoding="utf-8") as f:
                json.dump({"task": task, "item_id": item_id, "prompt": prompt,
                           "none": rows["none"]["plan"], "all": rows["task+user+global"]["plan"]},
                          f, ensure_ascii=False, indent=2, default=str)

            # 差异提示
            diff = set(rows["none"]["tools"]) != set(rows["task+user+global"]["tools"])
            print(f"\n>>> 记忆注入是否改变工具选择: {'✅ 有差异' if diff else '⚠️ 无差异'}")


if __name__ == "__main__":
    asyncio.run(main())
