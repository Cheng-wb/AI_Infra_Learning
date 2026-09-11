# distributed/

多 GPU 与集合通信实验（对应 M9）。

| 子目录 | 内容 |
|---|---|
| `nccl/` | topology 笔记、collective 数据流、nccl-tests benchmark、ring/tree 分析 |
| `tensor_parallel/` | column/row parallel、MLP/Attention TP、ReduceScatter+AllGather、通信计算 overlap |
| `expert_parallel/` | expert 分布、AllToAll、dispatch/combine、load imbalance |
| `nvshmem/` | PGAS model、put/get/signal、GPU-initiated communication demo |

目标：实现 Multi-GPU Transformer Block，产出 scaling + topology + profiler 报告。
