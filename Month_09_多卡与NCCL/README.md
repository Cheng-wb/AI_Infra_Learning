# Month 9：Multi-GPU — NCCL、TP/EP 与 NVSHMEM

> **月目标：** 从单卡 performance engineer 跨到真正的 distributed AI systems。
> **Must：** NCCL collectives、TP、ReduceScatter/AllGather、通信计算重叠。
> **Should：** Expert Parallel + AllToAll、拓扑意识。
> **Stretch：** NVSHMEM GPU-initiated communication mini project。

## Week 1 — Interconnect 与 NCCL Collectives

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Topology | PCIe/NVLink/NVSwitch/IB/RoCE 梳理 | 能画单机/多机拓扑 |
| 2 | Collective math | AllReduce/AllGather/ReduceScatter/AllToAll | 画数据流与字节量 |
| 3 | Ring AllReduce | 手算 ring 步骤/带宽模型 | 能解释 bandwidth-optimal 条件 |
| 4 | Tree/latency | 对比 ring/tree latency | 能解释小消息差异 |
| 5 | nccl-tests | 双卡/多卡跑 all_reduce_perf | 输出 size→GB/s |
| 6 | NCCL env/topology | 查看拓扑与关键调优变量 | 记录硬件限制 |
| 7 | 周收口 | collective benchmark report | 报告 bus bandwidth/algorithm bandwidth |

## Week 2 — Tensor Parallel

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Column parallel | 切 Linear 权重，局部 GEMM | 数学对拍 |
| 2 | Row parallel | 局部 GEMM + AllReduce/ReduceScatter | 正确重组输出 |
| 3 | MLP TP | 拼 column→activation→row | 与单卡 reference 对拍 |
| 4 | Attention TP | head sharding / projection sharding | forward 正确 |
| 5 | ReduceScatter+AllGather | 替代 naive AllReduce 路径 | 通信字节分析 |
| 6 | Overlap | communication stream + compute stream | nsys 看到 overlap |
| 7 | 周收口 | Multi-GPU Transformer block v0 | 2/4 GPU scaling 图 |

## Week 3 — Expert Parallel / AllToAll

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | EP decomposition | expert 按 rank 分布 | 画 dispatch/combine |
| 2 | AllToAll | 写 toy token exchange | correctness |
| 3 | Dispatch | routing→alltoall→expert | 端到端 toy MoE |
| 4 | Load imbalance | 构造 hotspot expert | 测 rank straggler |
| 5 | Overlap | dispatch 通信与局部工作重叠 | timeline 有证据 |
| 6 | TP+EP | 研究 vLLM/Megatron 组合 | 能解释 group 划分 |
| 7 | 周收口 | Distributed MoE note | latency breakdown |

## Week 4 — NVSHMEM 与 Multi-GPU Transformer Block

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | PGAS model | symmetric heap / PE / put/get | 能解释与 NCCL 的抽象差异 |
| 2 | GPU-initiated comm | 跑 NVSHMEM 最小示例 | kernel 内触发通信 |
| 3 | Put/Get/Signal | 双 GPU ping-pong 或 producer-consumer | 正确同步 |
| 4 | Fine-grained overlap | kernel 内计算+通信 toy pipeline | 与 host-orchestrated 对比 |
| 5 | Use case | attention/MoE 中找适合细粒度通信点 | 写设计文档 |
| 6 | 项目整合 | TP block + EP toy + benchmark | README 完整 |
| 7 | 月收口 | 发布 Multi-GPU Transformer Block | scaling + topology + profiler 报告 |

## Month 9 自检

- [ ] 能讲清 NCCL ring/tree、AllReduce/AllGather/ReduceScatter/AllToAll 的数据流与字节量。
- [ ] 跑过 nccl-tests，能读 bus bandwidth/algorithm bandwidth。
- [ ] 写过 column/row parallel + MLP/Attention TP，与单卡 reference 对拍，用 ReduceScatter+AllGather 替代 naive AllReduce。
- [ ] 用 communication stream + compute stream 实现 overlap，nsys 有证据。
- [ ] 理解 EP + AllToAll，能测 hotspot expert 的 rank straggler；跑过 NVSHMEM demo。
- [ ] Multi-GPU Transformer Block 已发布（scaling + topology + profiler 报告）。

> 完成后进入 [Month 10 — vLLM / Production Inference](../Month_10_vLLM与生产推理/README.md)。
