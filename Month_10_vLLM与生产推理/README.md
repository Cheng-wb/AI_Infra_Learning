# Month 10：vLLM / Production Inference Systems

> **月目标：** 学会分析「服务级性能」，而不是只盯一个 kernel。
> **Must：** vLLM 源码结构、scheduler/cache/worker、TTFT/TPOT/throughput。
> **Should：** disaggregated prefill/decode、DP/TP/EP、CUDA Graph、prefix caching。
> **Stretch：** 给真实 serving workload 做容量规划和性能回归系统。

## Week 1 — vLLM Source Deep Dive

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Engine architecture | 从 API 到 engine/worker 画调用链 | 模块图 |
| 2 | Scheduler | 跟一次请求的 schedule step | 写 trace note |
| 3 | KV cache manager | 对照 M6 mini runtime | 找设计差异 |
| 4 | Model runner | 跟 prefill/decode 执行 | 找 CUDA Graph/compile 接口 |
| 5 | Attention backend | FlashAttention/FlashInfer 等 backend | 能解释 backend selection |
| 6 | Quant/MoE kernel path | 定位 CUTLASS/CuTeDSL/quant path | 源码路径清楚 |
| 7 | 周收口 | vLLM architecture map | 附 10 个关键源码入口 |

## Week 2 — Scheduling / Parallelism / Disaggregation

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Chunked prefill | 改 token budget 做实验 | TTFT/TPOT 曲线 |
| 2 | Prefix cache | 重复/共享 prefix workload | cache hit 影响量化 |
| 3 | Speculative | 开启/对比一种 speculative mode | acceptance/latency 数据 |
| 4 | DP/TP | 部署 2/4 GPU 模式 | scaling 数据 |
| 5 | EP | MoE 模型或 toy EP 配置 | 理解 EP group |
| 6 | Disaggregated P/D | 阅读/运行可行 demo | 画 prefill/decode 数据流 |
| 7 | 周收口 | serving config playbook | 给不同 workload 推荐策略 |

## Week 3 — Production Performance Engineering

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Workload model | short/long prompt、interactive/batch 流量 | 生成 workload suite |
| 2 | Metrics | p50/p95 TTFT/TPOT/throughput | 自动采集 |
| 3 | Memory pressure | 改 KV cache size / max seq | 观察 OOM/eviction |
| 4 | Concurrency | 扫 request rate/concurrency | 找 saturation point |
| 5 | nsys trace | 服务端完整 trace | CPU/GPU bubble 定位 |
| 6 | Regression | 保存基线与阈值 | 自动判定性能回归 |
| 7 | 周收口 | production benchmark harness | 可重复跑不同版本 |

## Week 4 — P7 Production Inference Report

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Capacity planning | 根据显存/吞吐估单卡容量 | 有计算模型 |
| 2 | Bottleneck taxonomy | CPU launch / kernel / memory / comm / scheduling | 建诊断树 |
| 3 | Optimization | 选一个真实瓶颈改配置/代码 | 有 before/after |
| 4 | Reliability | timeout/cancel/OOM recovery 思路 | 写故障测试 |
| 5 | Deployment | Docker/版本锁定/复现脚本 | 环境可重建 |
| 6 | Report | 写 10–15 页等价深度性能报告 | 数据/结论完整 |
| 7 | 月收口 | 发布 P7 + 更新简历 v2 | 至少一个 production 级案例 |

## Month 10 自检

- [ ] 能画出 vLLM 架构图（engine/scheduler/KV cache manager/model runner/attention backend），附 10 个关键源码入口。
- [ ] 能对比 chunked prefill、prefix cache、speculative、DP/TP/EP、disaggregated P/D 的 trade-off，形成 serving config playbook。
- [ ] 有 workload suite + p50/p95 TTFT/TPOT/throughput 自动采集，能定位 OOM/eviction/saturation。
- [ ] 用 nsys 做服务端完整 trace，定位 CPU/GPU bubble，建立性能回归基线。
- [ ] P7 Production Inference Performance Report 已发布（容量模型 + 诊断树 + before/after + 部署脚本），简历更新到 v2。

> 完成后进入 [Month 11 — 开源深挖 / 专项强化](../Month_11_开源深挖与专项/README.md)。
