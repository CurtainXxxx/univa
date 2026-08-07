# eval/ 评测目录

UniVA-Bench 实验执行器（runner.py）+ 指标（metrics.py）+ 运行产物。

## 目录结构

```
eval/
├── runner.py              # 实验执行器（--mode agent | single | direct）
├── metrics.py             # CLIP / DINO / MLLM Judge 指标
├── mcp_configs_eval.json  # 评测用 MCP 配置（5 个 server）
├── README.md              # 本文件
├── outputs/               # 各次 run 的结果（gitignore，本地保留）
└── tmp/                   # 临时调试文件（gitignore，随时可清）
```

## outputs/ 各 run 说明

| run 目录 | 用途 | 口径 | 结果 |
|---|---|---|---|
| `smoke1/` | 早期冒烟（lt2v/it2v/lvu 各 1 条） | direct | 链路验证用，失败因当时 bug |
| `lvu_batch1/` | lvu 评测第一版（10 条） | **32 帧 / 只测第 1 问** | 6/10 = **60%**（历史口径，已弃） |
| `lvu_batch2/` | lvu 正式评测·前 6 条 | 64 帧 / 全 QA | 中断残留（余额不足 + 简答无 options bug） |
| `lvu_batch3/` | lvu 正式评测·后 4 条 | 64 帧 / 全 QA + 简答 LLM judge | 4 条含 3MJNKd10kxM（70%） |
| `single_test1/` | SingleAgent 首测（lt2v 1 条） | single 模式 | 1 个 5s 付费视频（`lt_0_generated.mp4`）+ 重复提交教训 |

## lvu 正式结果（batch2 + batch3 合并）

**10 视频 / 103 题 = 74.8%**（按题目），74.7%（按视频）。
对比：论文 InternVL3-38B = 75%，UniVA = 76%。

- 口径：64 帧均匀采样、全部 QA（非抽样）、选择题字母匹配 + 简答题 LLM judge
- 关键教训：评测口径（全 QA vs 第 1 问）+ 帧数（64 vs 32）直接决定结论，这是复现论文最容易踩的坑

## 运行方式（充值后）

```bash
# 环境：必须用 venv + PATH 前缀（MCP 子进程同环境）
export PATH="/d/univa/venv/Scripts:$PATH"

# Plan-Act 对照
python -m eval.runner --tasks lt2v,it2v --items 0,1 --mode agent \
  --mcp-config eval/mcp_configs_eval.json --run-id cmp_planact

# Single-Agent 对照
python -m eval.runner --tasks lt2v,it2v --items 0,1 --mode single \
  --mcp-config eval/mcp_configs_eval.json --run-id cmp_single
```

每个 item 完成即写 `<id>.json` manifest，执行中写 `<id>.pending.json` 标记（中断可查进度）。
