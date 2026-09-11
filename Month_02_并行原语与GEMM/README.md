# Month 2：并行原语、GEMM 与 Roofline

> **月目标：** 真正掌握「数据复用 → 并行分解 → 性能模型 → profiler」的优化方法。
> **Must：** Scan + GEMM naive/shared/register-tiled；Roofline。
> **Should：** cp.async/double buffering。
> **Stretch：** 对 LLM 常见 GEMM shape 做系统 autotune。

## Week 1 — Scan / Histogram / Prefix Primitives

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Scan 定义 | CPU inclusive/exclusive + Hillis-Steele | reference 完整 |
| 2 | Blelloch scan | 单 block up-sweep/down-sweep | work-efficient、对拍 |
| 3 | Multi-block scan | block sum + offset scan + add back | 任意 N 通过 |
| 4 | Stream compaction | mask → scan → scatter | 与 CPU 结果一致 |
| 5 | Histogram / atomics | global atomic vs shared privatization | 解释 contention |
| 6 | Warp primitives | ballot/shuffle/activemask 小练习 | 能写 warp-level 协作 |
| 7 | 周收口 | primitives benchmark | 输出 size→latency/bandwidth 曲线 |

## Week 2 — GEMM Baseline + Roofline

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | GEMM 数学与 shape | CPU SGEMM + FLOPs/bytes 计算 | 能算算术强度 |
| 2 | Naive CUDA GEMM | 每线程一个 C 元素 | 对拍 cuBLAS |
| 3 | Roofline | 建 4090 实测/理论 Roofline | 把 naive kernel 放到图上 |
| 4 | Shared tiling | BM/BN/BK tile GEMM | 明显快于 naive |
| 5 | Tile 参数 | 扫 tile 组合与 occupancy | 输出参数热力表 |
| 6 | cuBLAS baseline | 同 shape/dtype/layout 对比 | benchmark 口径一致 |
| 7 | 周收口 | 写「为什么 tiled GEMM 快」 | 数据复用可定量说明 |

## Week 3 — Register Tiling / Vectorization / Async

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Register tiling | 每线程计算 TM×TN micro-tile | profiler 看算术/访存变化 |
| 2 | Vectorized load | float4 载入 A/B | 对齐与 tail 正确 |
| 3 | Double buffer | shared double-buffer pipeline | 能解释 overlap |
| 4 | cp.async | 在支持架构上改 tile load | 比较同步加载与异步路径 |
| 5 | Unroll / ILP | K-loop unroll，观察寄存器压力 | 找到性能拐点 |
| 6 | SASS 初识 | cuobjdump/nvdisasm 看 load/FMA | 对照代码识别关键指令 |
| 7 | 周收口 | GEMM 优化阶梯图 | 每一步都有数据 |

## Week 4 — LLM GEMM Shapes 与 P2

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Shape 分类 | 方阵、skinny、fat、小 M decode shape | 建 shape suite |
| 2 | Shape-specific tuning | 对 3 类 shape 分别调 tile | 不追求一个配置打天下 |
| 3 | Batched GEMM | strided batched GEMM 基础 | 与 cuBLAS 对拍 |
| 4 | Epilogue | bias + activation 融合 | 减少中间写回 |
| 5 | Nsight 深入 | 选最差 shape 找 bottleneck | 形成证据链 |
| 6 | P2 文档 | benchmark protocol + profiler 截图/导出 | 陌生人可复现 |
| 7 | 月收口 | 发布 P2 + GEMM 博客 | 给出「在哪些 shape 快/慢」 |

## Month 2 自检

- [ ] 手写 Hillis-Steele / Blelloch scan + 多 block scan，任意 N 对拍通过。
- [ ] 会算 GEMM 的 FLOPs/算术强度，能建 4090 Roofline 并定位 naive kernel。
- [ ] 手写 naive → shared tiling → register tiling GEMM，每步有 benchmark 数据。
- [ ] 会用 float4 向量化、double buffering、cp.async，能解释 overlap。
- [ ] 能看 cuobjdump/nvdisasm 的 load/FMA 指令，对照源码。
- [ ] P2 CUDA GEMM 优化报告已发布（benchmark protocol + profiler 证据 + 复现步骤）。

> 完成后进入 [Month 3 — Tensor Core / CUTLASS / CuTe](../Month_03_TensorCore与CUTLASS/README.md)。
