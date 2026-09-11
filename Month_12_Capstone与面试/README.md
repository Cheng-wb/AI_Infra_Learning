# Month 12：Capstone Final、面试与正式投递

> **月目标：** 技术继续占约 50%，另外 50% 用于面试表达、手撕、系统设计与正式投递。
> **Must：** Capstone 最终版、强简历、kernel/system design 面试准备。
> **Should：** 真实面试驱动最后补漏。
> **Stretch：** 拿到目标岗位 offer 或进入高质量终面。

## Week 1 — Capstone Final Benchmark

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Benchmark freeze | 固定版本/环境/shape suite | baseline 锁定 |
| 2 | Cross-implementation | CUDA/Triton/CuTe/cuTile 中选 2–3 类对比 | 说明选择依据 |
| 3 | Cross-architecture | 4090 + 可获得的 H100/B200 | 有跨卡数据或清楚限制 |
| 4 | Runtime benchmark | TTFT/TPOT/tok/s/concurrency | 服务指标完整 |
| 5 | Distributed benchmark | 2/4 GPU scaling（可行时） | efficiency 报告 |
| 6 | Final profiling | ncu + nsys 形成证据链 | 一页性能故事 |
| 7 | 周收口 | Capstone v1.0 release | README/CI/bench 全齐 |

## Week 2 — Kernel / C++ / Compiler 面试

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | 手撕基础 kernel | reduction/transpose/softmax | 30–45 min 内写出 |
| 2 | 手撕 GEMM | tiled GEMM 伪码 + 优化解释 | 能推 tile/occupancy |
| 3 | Attention | FlashAttention/PagedAttention 白板 | 数据流清楚 |
| 4 | C++ | RAII/template/memory model/concurrency | 常见题有代码例子 |
| 5 | Compiler | torch.compile/Triton/IR/PTX/SASS | 能讲 lowering chain |
| 6 | Debug case | 给一个慢 kernel 做现场诊断 | 有系统 checklist |
| 7 | 周收口 | Mock interview #1 | 按失分点补漏 |

## Week 3 — Systems / Distributed / Serving 面试 + 正式投递

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Serving design | 设计 LLM inference engine | scheduler/cache/metrics 齐全 |
| 2 | Distributed design | 设计 8 GPU TP/EP serving | topology/collective 清楚 |
| 3 | Performance design | 如何把吞吐提高 30% | 先测量再优化 |
| 4 | Project storytelling | GEMM / runtime / PR 三个 5 分钟案例 | 数字可信 |
| 5 | Resume tailoring | 针对 5 个 JD 改关键词与 bullet | 每份有针对性 |
| 6 | Formal applications | 集中内推/官网/社区投递 | 建 tracking sheet |
| 7 | 周收口 | Mock interview #2 + 复盘 | 更新高频错题 |

## Week 4 — Feedback Loop 与毕业验收

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | 面试反馈 | 汇总不会/讲不清/证据弱 | 排 P0/P1 缺口 |
| 2 | P0 补漏 | 只补最影响 offer 的问题 | 一天解决一个主缺口 |
| 3 | Portfolio | 一页索引：项目/PR/博客/bench | 面试官 5 分钟能看懂 |
| 4 | Open-source follow-up | 跟 PR/reviewer | 保持活跃 |
| 5 | 技术总结 | 写《12 个月 AI Kernel/Systems 复盘》 | 公开或私有完整文档 |
| 6 | 90-day plan | 写入职后 90 天计划 | 与目标岗对齐 |
| 7 | 最终验收 | 对根 README 第 4 节清单逐项打勾 | 明确通过/下一阶段 |

## Month 12 自检

- [ ] Capstone v1.0 release（README/CI/bench 全齐，跨实现/跨架构/服务指标/多卡 scaling/证据链齐备）。
- [ ] 手撕基础 kernel / tiled GEMM 能在 30–45 min 内写出，能白板讲 FlashAttention/PagedAttention。
- [ ] kernel/system design 面试可讲：serving design、8 GPU TP/EP、吞吐提升 30% 的方法论。
- [ ] 针对 5 个 JD 定制简历，建投递 tracking sheet，完成 2 次 mock interview 并复盘。
- [ ] 汇总面试反馈、补 P0 缺口、整理一页 portfolio、写《12 个月复盘》与 90 天入职计划。
- [ ] 对照最终验收清单逐项打勾，明确通过/下一阶段。

> 全部完成后，回到 [根 README 第 4 节](../README.md) 做最终验收，并规划下一步（入职 90 天 / 长期 3 年）。
