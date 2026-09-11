# Month 3：Tensor Core、CUTLASS 与 CuTe

> **月目标：** 从手写 FP32 GEMM 进入 Tensor Core 与工业 kernel abstraction。
> **Must：** FP16/BF16 Tensor Core、CUTLASS GEMM、CuTe layout 基础。
> **Should：** CuTe DSL 写一个 GEMM。
> **Stretch：** Hopper/Blackwell 上运行对应 DSL/CUTLASS 示例。

## Week 1 — Mixed Precision 与 Tensor Core

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | FP16/BF16/TF32 | 数值范围/精度实验 | 能解释 accumulation dtype |
| 2 | Tensor Core 模型 | 跑 WMMA 最小示例 | 对拍 cublasGemmEx |
| 3 | WMMA GEMM | tiled WMMA kernel | 正确处理边界 |
| 4 | Shared → MMA | shared staging + fragments | 理解数据路径 |
| 5 | ldmatrix/mma 概念 | 阅读 PTX 并定位 MMA 指令 | 能从 SASS/PTX 找 tensor op |
| 6 | 精度与吞吐 | FP32/TF32/FP16/BF16 对比 | 同时报告误差与速度 |
| 7 | 周收口 | Tensor Core benchmark | 写数据类型选择指南 |

## Week 2 — CUTLASS C++

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | CUTLASS 架构 | 编译示例，画 GEMM hierarchy | threadblock/warp/instruction 层清楚 |
| 2 | Mainloop | 读一个 Ampere/Ada GEMM mainloop | 画 load→mma pipeline |
| 3 | Epilogue | 改 bias/relu epilogue | 正确 + benchmark |
| 4 | Layout | Row/Column/stride/layout 变化 | 能解释 layout 约束 |
| 5 | Kernel selection | 不同 tile/stage 参数 | 输出性能差异 |
| 6 | CUTLASS profiler | 用 profiler 扫 shape | 找到优选 kernel |
| 7 | 周收口 | 写 CUTLASS 快速定位笔记 | 能从入口追到 mainloop/epilogue |

## Week 3 — CuTe 与 CuTe DSL

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | CuTe Layout Algebra | shape/stride/layout 手算 | 能解释 coordinate→offset |
| 2 | Tensor / TiledCopy | 做 layout/copy 小例子 | 能画 thread-data mapping |
| 3 | MMA atom | 阅读/运行 CuTe MMA 示例 | 理解 atom→tiled MMA |
| 4 | CuTe DSL 环境 | 跑 vector/GEMM 教程 | JIT 成功 |
| 5 | CuTe DSL GEMM | 修改 shape/layout/stage | correctness + baseline |
| 6 | Profiling DSL | profiler/autotune/JIT cache | 记录 compile/run 时间 |
| 7 | 周收口 | CUDA/CUTLASS/CuTe DSL 对照表 | 说明控制力与开发效率 |

## Week 4 — Hopper / Blackwell 编程模型预热

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Hopper：TMA | 读官方 TMA + CUTLASS sm90 path | 能解释为何减少线程搬运 |
| 2 | Hopper：warpgroup MMA | 找 wgmma pipeline | 能解释 warp→warpgroup 变化 |
| 3 | Thread Block Cluster | cluster/DSMEM 示例 | 能解释跨 block 协作 |
| 4 | Blackwell compute 10.x | 读 tuning/programming guide | 记录与 Hopper 的关键差异 |
| 5 | Architecture targets | `sm_89/sm_90/sm_100*` 编译概念 | 理解 baseline/family/arch target |
| 6 | 云 GPU 实测 | 若可租 H100/B200，跑 CUTLASS/CuTe 示例 | 有一组跨架构数字；否则完成编译/源码报告 |
| 7 | 月收口 | P3：多实现 GEMM + 架构笔记 | README 明确硬件适用范围 |

## Month 3 自检

- [ ] 能解释 FP16/BF16/TF32 的数值范围与 accumulation dtype，跑通 WMMA GEMM 并对拍 cublasGemmEx。
- [ ] 能从 PTX/SASS 定位 MMA/tensor op 指令。
- [ ] 会用 CUTLASS 跑 GEMM，改 epilogue/layout/tile，能从入口追到 mainloop/epilogue。
- [ ] 理解 CuTe Layout Algebra（coordinate→offset），能用 CuTe DSL 写并改 GEMM。
- [ ] 能解释 Hopper TMA/wgmma/cluster 与 Blackwell compute 10.x 的关键差异。
- [ ] P3 多层次 GEMM + CuTe/CUTLASS 笔记已发布（README 明确硬件适用范围）。

> 完成后进入 [Month 4 — DL 核心算子 / Triton / FlashAttention](../Month_04_LLM算子与Triton/README.md)。
