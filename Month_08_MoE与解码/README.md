# Month 8：MoE、Sampling、Speculative Decoding 与开源贡献

> **月目标：** 学现代推理系统中 FlashAttention 之外的热点路径。
> **Must：** MoE routing、permute、grouped GEMM；sampling。
> **Should：** speculative decoding。
> **Stretch：** 对 vLLM/FlashInfer/SGLang 提交一个 kernel/runtime PR。

## Week 1 — MoE Routing

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | MoE execution | top-k router + expert mapping reference | 能画 token→expert 流程 |
| 2 | Top-k | CUDA/Triton top-k 小 kernel | correctness |
| 3 | Prefix/count | expert histogram + offsets | 无原子热点或能解释热点 |
| 4 | Permute | token reorder by expert | 对拍 CPU |
| 5 | Capacity/load | 构造偏斜 routing | 测负载不均衡 |
| 6 | Unpermute/combine | expert output 回排 | 完整 forward dataflow |
| 7 | 周收口 | MoE routing kernel pack | benchmark + 数据流图 |

## Week 2 — Grouped GEMM / Fused Experts

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Grouped GEMM | 多 expert 不同 M 的 GEMM reference | shape suite |
| 2 | CUTLASS/CuTe grouped GEMM | 跑官方/示例路径 | baseline 数据 |
| 3 | Small-M optimization | decode expert GEMM 分析 | 找 launch/occupancy 问题 |
| 4 | Fused activation | expert GEMM + SiLU/GELU | 减少 kernel/memory traffic |
| 5 | Routing+GEMM pipeline | 串联 permute→GEMM→unpermute | correctness |
| 6 | Profiling | nsys/ncu 看 MoE timeline | 找 top bottleneck |
| 7 | 周收口 | MoE mini-engine v1 | throughput 曲线 |

## Week 3 — Sampling / Speculative Decoding

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Sampling math | temperature/top-k/top-p reference | 分布验证 |
| 2 | GPU top-k | kernel 化 top-k | 与 torch baseline 比较 |
| 3 | Top-p | sort/scan/filter 路径分析 | correctness |
| 4 | Fused sampling | logits transform + sampling 融合 | 减少中间写回 |
| 5 | Speculative decoding | draft/verify 算法模拟 | 能解释 acceptance |
| 6 | Verify kernel | batch verify toy implementation | 输出一致 |
| 7 | 周收口 | decoding optimization note | 指出何时收益/何时无收益 |

## Week 4 — 开源贡献 Sprint 1

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | 目标项目 | 选 vLLM/FlashInfer/SGLang/CUTLASS/Triton | 本地测试跑通 |
| 2 | Hot path | 用 issue + profiler 找具体问题 | 写 issue analysis |
| 3 | Reproduce | 建最小 benchmark/reproducer | 问题稳定复现 |
| 4 | Patch | 修 bug/性能/测试 | 本地数据支持 |
| 5 | PR | 写完整 PR 描述和 benchmark | CI 进入 review |
| 6 | Review iteration | 回应 reviewer / 补测试 | PR 质量提升 |
| 7 | 月收口 | MoE 项目 + PR + 博客 | merge 是加分，不是唯一 KPI |

## Month 8 自检

- [ ] 手写 MoE routing（top-k、histogram+offset、permute/unpermute），完整 forward dataflow 对拍。
- [ ] 能跑 grouped GEMM（CUTLASS/CuTe），做 small-M 优化与 fused activation，MoE mini-engine v1 有 throughput 曲线。
- [ ] 手写 sampling（temperature/top-k/top-p）+ GPU top-k，能解释 speculative decoding 的 acceptance。
- [ ] 提交第一个有技术含量的 PR（issue analysis → reproducer → patch → review）。
- [ ] MoE mini-engine + PR + 博客已产出。

> 完成后进入 [Month 9 — NCCL / TP/EP / NVSHMEM](../Month_09_多卡与NCCL/README.md)。
