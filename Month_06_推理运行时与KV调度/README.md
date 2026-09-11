# Month 6：LLM Inference Runtime — KV Cache、PagedAttention 与 Scheduler

> **月目标：** 从「会优化单个 kernel」升级到「理解为什么整个 LLM serving 系统快或慢」。
> **Must：** prefill/decode、KV Cache manager、PagedAttention 思想、continuous batching。
> **Should：** chunked prefill、prefix caching、CUDA Graph。
> **Stretch：** Mini Runtime 能接一个小型真实模型。
> **求职动作：** 本月开始小规模试投，不等到 M10。

## Week 1 — Prefill / Decode / KV Cache

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | LLM execution anatomy | 拆 Transformer prefill/decode kernel 时间 | timeline 图 |
| 2 | Serving metrics | 实现 TTFT/TPOT/tok/s 统计 | 指标定义正确 |
| 3 | KV Cache 基础 | 写 contiguous KV cache manager | allocate/append/free 测试 |
| 4 | KV shape/layout | MHA/GQA 的 cache shape | 能算显存占用 |
| 5 | Decode attention | 从 cache 读取单 token attention | correctness |
| 6 | Memory-bound decode | Roofline/带宽分析 decode | 能解释 batch 对性能影响 |
| 7 | 周收口 | KV Cache v0 + 性能笔记 | 有 cache capacity 计算器 |

## Week 2 — Paged KV Cache / PagedAttention

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Paging 思想 | page/block table + logical→physical 映射 | 画映射图 |
| 2 | Block allocator | free list / allocate / release | 随机测试无泄漏 |
| 3 | Paged KV write | token 写入非连续 block | 对拍 contiguous cache |
| 4 | Paged read | attention 按 block table gather K/V | correctness |
| 5 | Fragmentation | 构造不同请求长度，比较浪费 | 量化内存利用率 |
| 6 | Kernel 优化 | 减少 indirection/gather 开销 | benchmark paged read |
| 7 | 周收口 | PagedAttention mini prototype | tests + memory report |

## Week 3 — Continuous Batching / Chunked Prefill / Prefix Cache

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Request state machine | waiting/running/finished | 单测状态转换 |
| 2 | Continuous batching | 每步动态加入/退出请求 | scheduler 可跑模拟流量 |
| 3 | Token budget | 实现 max batched tokens 约束 | 不超 cache/token budget |
| 4 | Chunked prefill | 长 prompt 拆 chunk 与 decode 共批 | TTFT/TPOT trade-off 图 |
| 5 | Prefix caching | hash prefix→reuse KV blocks | 重复 prompt 命中 |
| 6 | Eviction policy | LRU/引用计数基础 | cache pressure 测试 |
| 7 | 周收口 | scheduler v1 | trace 记录每步决策 |

## Week 4 — Mini LLM Runtime v1 + 试投

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | CUDA Graph | 固定 decode shape capture/replay | 小 kernel launch overhead 降低 |
| 2 | Runtime 集成 | scheduler + cache + attention | toy generation loop 跑通 |
| 3 | vLLM baseline | 同模型/近似配置跑 vLLM | 指标同口径 |
| 4 | Profiling | nsys 看 runtime 时间线 | 找 top 3 系统瓶颈 |
| 5 | P6 报告 | TTFT/TPOT/throughput/memory | 报告可复现 |
| 6 | 简历 v1 + 试投 | 写 3 个项目 bullet，投少量目标岗 | 收集 JD/面试反馈 |
| 7 | 月收口 | 发布 Mini Runtime v1 | M6 checkpoint 完成 |

## Month 6 自检

- [ ] 能解释 prefill/decode 的性能差异（compute vs memory-bound），会用 TTFT/TPOT/tok/s。
- [ ] 写过 contiguous KV cache manager 与简化 PagedAttention（block allocator + block table gather）。
- [ ] 写过 continuous batching scheduler，含 token budget、chunked prefill、prefix caching、LRU eviction。
- [ ] 用 CUDA Graph 降低 decode launch overhead，跑通 toy generation loop，用 nsys 找 top 3 瓶颈。
- [ ] P6 Mini LLM Inference Runtime v1 已发布（TTFT/TPOT/throughput/memory 可复现报告）。

### M6 Checkpoint

到这里应当已经具备第一轮 AI Kernel / AI Systems 岗位投递能力。最低证据：P2 GEMM、P4 LLM kernels、P5 PyTorch integration、P6 Mini Runtime，至少 2 篇高质量技术文章，并开始真实获取市场反馈。

> 完成后进入 [Month 7 — FP8/INT8/FP4 / cuTile / Blackwell](../Month_07_低精度与cuTile/README.md)。
