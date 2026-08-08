"""
Success Rate 统计（论文 5.2 Agentic Probing 核心指标）

论文定义：Success Rate = wPED > 0 的测试比例（即"计划结构性有效"的比例，
            能捕捉空计划 / 畸形输出等灾难性失败）。
本工具用两个代理判定替代 wPED>0（参考计划不可得，见 README）：

  1. 计划有效性  plan_valid   — execution_plan 能解析、steps 非空
  2. 执行成功率  exec_success — 至少一个 step 成功且产出有效
                                （生成类看 output_path，理解类看答案）

用法：
  python -m eval.success_rate --run cmp_planact --run cmp_single
  或对比两个 run：
  python -m eval.success_rate --compare cmp_planact cmp_single

输出：每个 run 的 Success Rate + 计划有效性，可选对比表。
"""

import argparse
import json
import os
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "outputs"


def analyze_run(run_name: str) -> dict:
    """统计一个 run 目录下所有 manifest 的 Success Rate。"""
    run_dir = OUT_ROOT / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    results = {}
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for f in sorted(task_dir.glob("*.json")):
            if f.name.endswith(".pending.json") or f.name == "summary.json":
                continue
            item_id = f.stem
            m = json.load(open(f, encoding="utf-8"))
            results[f"{task}/{item_id}"] = analyze_manifest(m)

    return {
        "run": run_name,
        "total": len(results),
        "plan_valid": sum(1 for r in results.values() if r["plan_valid"]),
        "exec_success": sum(1 for r in results.values() if r["exec_success"]),
        "items": results,
    }


def analyze_manifest(m: dict) -> dict:
    """对单个 manifest 判定计划有效性 + 执行成功率。"""
    # 1) 计划有效性：能从 plan/execution 还原出结构化计划
    plan = m.get("plan")
    execution = m.get("execution")
    plan_valid = False
    steps = 0

    if isinstance(plan, dict) and "execution_plan" in plan:
        plan_steps = plan.get("execution_plan", {}).get("steps", [])
        steps = len(plan_steps)
        plan_valid = steps > 0
    elif isinstance(execution, dict) and execution:
        steps = len(execution)
        plan_valid = True  # 有执行步骤记录，视为计划已生成

    # 2) 执行成功率：任一 step 成功且产出有效
    exec_success = False
    if isinstance(execution, dict):
        for step in execution.values():
            if not isinstance(step, dict):
                continue
            ok = str(step.get("success")).lower() in ("true", "success")
            output = step.get("output_path") or step.get("message")
            if ok and output:
                exec_success = True
                break
    else:
        # 兼容旧 manifest：顶层 success 字段
        exec_success = m.get("success") in (True, "True", "true")

    return {"plan_valid": plan_valid, "exec_success": exec_success, "steps": steps}


def format_run(r: dict) -> str:
    lines = [
        f"=== {r['run']} ===",
        f"  总任务   : {r['total']}",
        f"  计划有效 : {r['plan_valid']}/{r['total']} ({r['plan_valid']/max(r['total'],1):.0%})",
        f"  执行成功 : {r['exec_success']}/{r['total']} ({r['exec_success']/max(r['total'],1):.0%})",
    ]
    for k, v in r["items"].items():
        mark = "OK" if v["exec_success"] else "--"
        pmark = "V" if v["plan_valid"] else "X"
        lines.append(f"    [{pmark}][{mark}] {k} (steps={v['steps']})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", help="要统计的 run 名，可多个")
    parser.add_argument("--compare", nargs=2, help="对比两个 run（如 cmp_planact cmp_single）")
    args = parser.parse_args()

    runs = []
    if args.compare:
        runs = [analyze_run(x) for x in args.compare]
        r1, r2 = runs
        print(f"Success Rate:  {r1['run']}={r1['exec_success']/max(r1['total'],1):.1%}  vs  "
              f"{r2['run']}={r2['exec_success']/max(r2['total'],1):.1%}")
        print(f"计划有效性:    {r1['run']}={r1['plan_valid']/max(r1['total'],1):.1%}  vs  "
              f"{r2['run']}={r2['plan_valid']/max(r2['total'],1):.1%}")
    elif args.run:
        runs = [analyze_run(x) for x in args.run]

    for r in runs:
        print()
        print(format_run(r))


if __name__ == "__main__":
    main()
