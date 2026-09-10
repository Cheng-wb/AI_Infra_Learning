# Month 4：Tensor Core 与进阶算子

> **主题：** 用 Tensor Core 加速矩阵运算（WMMA / CUTLASS），实现深度学习核心算子 Softmax / LayerNorm / Attention。
> **前置：** 完成 M3（能写 register-tiled GEMM、理解 roofline）。
> **月度目标：** 用 WMMA 写出比 FP32 普通 GEMM 快数倍的 GEMM，掌握 CUTLASS；完成 Softmax / LayerNorm / Attention 算子；完成 **P3 Tensor Core 算子集**。

## 月度里程碑

- ✅ 理解 Tensor Core 的矩阵分片（m16n8k16 / m16n8k8）与混合精度（FP16 输入、FP32 累加）。
- ✅ 用 WMMA API 写出 Tensor Core GEMM，对比 FP32 GEMM 加速比。
- ✅ 会用 CUTLASS 跑 GEMM 并理解其 tile 抽象。
- ✅ 实现数值稳定的 Softmax（online/max 减法）、LayerNorm、常见激活。
- ✅ 实现分块 Attention（naive → 分块），为 Flash Attention 打基础。
- ✅ 完成 P3：Tensor Core GEMM + 三算子 + benchmark。

---

## Week 1：Tensor Core 原理与 WMMA

**周目标：** 理解 Tensor Core 如何在一个周期内完成 4×4×4 矩阵乘累加，会用 WMMA API 写 kernel。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Tensor Core 原理 | Tensor Core 是专用矩阵乘累加单元，`D = A*B + C`；FP16 输入 / FP32 累加；4090 有第 4 代 Tensor Core（Ada） | 读 NVIDIA 博客《Programming Tensor Cores in CUDA 9》 | 能解释 Tensor Core 与普通 CUDA core 的区别 |
| Day 2 | WMMA 数据类型 | `nvcuda::wmma::fragment`、`matrix_a/b/accumulator`、`load_matrix_sync`、`mma_sync`、`store_matrix_sync` | 抄写官方 WMMA 最小示例并跑通 | 理解 fragment 与共享内存/寄存器的映射 |
| Day 3 | WMMA GEMM | 用 WMMA 写一个 tile 级 GEMM：`load` 16×16 分片 → `mma` → `store` | 写 WMMA GEMM（单 tile），对拍 | 跑通，理解 m16n16k16 分片 |
| Day 4 | 完整 WMMA GEMM | 外层 tile 循环 + 内层 WMMA，把 FP16 GEMM 拼完整 | 完成完整 WMMA GEMM，与 FP32 GEMM 对比 | 看到数倍加速，记录数字 |
| Day 5 | 混合精度与精度损失 | FP16 表示范围、溢出/舍入、`__half`/`half2`、累加精度 | 对比 FP32 GEMM 与 FP16 WMMA GEMM 的误差 | 能解释精度损失的来源与影响 |
| Day 6 | WMMA 与共享内存 | fragment 从共享内存加载，tile 复用 + 双缓冲 | 把 WMMA GEMM 的输入改成共享内存 tile | 性能进一步提升 |
| Day 7 | 复习 + 综合 | WMMA GEMM vs FP32 GEMM vs cuBLAS（`cublasGemmEx` FP16）三对比 | 完成 benchmark + 笔记 | 交出 WMMA GEMM 对比笔记 |

**Week 1 自检：** 能解释 Tensor Core 的分片与混合精度；能手写 WMMA GEMM；能对比 FP32/FP16 速度与精度。

---

## Week 2：CUTLASS 框架

**周目标：** 掌握 CUTLASS 的 tile 抽象与模板元编程思想，能用它跑 GEMM 并理解底层。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | CUTLASS 概览 | CUTLASS 是 NVIDIA 开源的高性能模板库，用「tile / threadblock / warp / instruction」分层抽象 | 拉取 CUTLASS，编译官方 example | 环境跑通 |
| Day 2 | Tile 抽象 | `ThreadblockShape`、`WarpShape`、`InstructionShape` 三者关系 | 读 `examples/00_basic_gemm`，跑不同 tile 组合 | 理解三层 tile 的层级 |
| Day 3 | 模板元编程 | `using` 类型别名、`traits`、编译期参数选择 | 改 example 里的模板参数（tile 大小、数据类型） | 能改参数并观察性能变化 |
| Day 4 | 混合精度 GEMM | CUTLASS 的 FP16/INT8 GEMM 路径 | 跑 `basic_gemm` 的 FP16/INT8 变体 | 记录各精度 GFLOPs |
| Day 5 | 与自写 GEMM 对比 | CUTLASS vs 自己 M3 的 GEMM vs cuBLAS | 三者同规模 benchmark | 理解 CUTLASS 为什么快 |
| Day 6 | 读核心源码 | 精读一个 CUTLASS GEMM 的 mainloop（`mma` 调用、`ldmatrix`、`cp.async`） | 跟着源码画数据流图 | 能讲清 CUTLASS GEMM 的执行流程 |
| Day 7 | 复习 + 综合 | 整理 CUTLASS 笔记（分层抽象 + 关键 API） | 完成 Week 2 笔记 | 交出 CUTLASS 笔记（M8 深挖的底子） |

**Week 2 自检：** 能用 CUTLASS 跑 GEMM 并改参数；能画 CUTLASS 的 tile 层级；能说清它比自写 GEMM 快的原因。

---

## Week 3：Softmax / LayerNorm / 激活算子

**周目标：** 实现数值稳定的 Softmax 与 LayerNorm，掌握 warp 级原语与在线算法。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 数值稳定 Softmax | `softmax(x)_i = e^{x_i - max}/Σe^{x_j - max}`，减 max 防溢出 | 手写 CPU 稳定 softmax，对比 naive 版在 `x=1000` 时是否 NaN | 能解释「减 max」为什么必要 |
| Day 2 | CUDA Softmax（分块归约） | 每行一个 block：先归约 max、再归约 sum、再归一 | 写行 softmax，复用 M2 的归约，对拍 | 跑通 + 对拍 |
| Day 3 | Online Softmax | 一遍遍历同时更新 max 与 sum，省一次归约 | 写 online softmax（warp shuffle 版），对拍 | 理解「在线更新」如何减少遍历 |
| Day 4 | LayerNorm | `y = (x - mean)/sqrt(var + eps) * γ + β`，先归约求 mean/var | 写 LayerNorm kernel + 反向梯度（M5 用），对拍 | 能写前向 + 反向 |
| Day 5 | 激活算子 | relu/gelu/silu/swish 的公式与 CUDA 实现，`tanh`/`erf` 精度 | 写激活算子集，对拍 PyTorch | 算子集通过 |
| Day 6 | warp 原语进阶 | `__ballot_sync`、`__shfl_sync`、`__activemask`、`__syncwarp` | 写一个用 ballot 做条件判断的小例子 | 会用 warp 级协作原语 |
| Day 7 | 复习 + 综合 | Softmax/LayerNorm/激活三算子收进算子库 + benchmark | 完成 Week 3 算子 + 笔记 | 交出三算子（P3 一部分） |

**Week 3 自检：** 能手写稳定 softmax 与 online softmax；能手写 LayerNorm 前向+反向；能用 warp shuffle 做归约。

---

## Week 4：Attention 基础

**周目标：** 实现 self-attention 的朴素版与分块版，理解显存/访存瓶颈，为 M5 Flash Attention 打基础。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Attention 公式 | `Attention(Q,K,V) = softmax(QK^T/√d)V`；Q/K/V 形状；缩放因子 | 手写 CPU attention，与 PyTorch 对拍 | 能写清 QK^T → scale → softmax → @V 流程 |
| Day 2 | Naive Attention | 直接算出 `S = QK^T`（N×N），再 softmax，再乘 V；N 大时显存爆炸 | 写 naive CUDA attention，对拍 | 理解「N×N 中间矩阵」是瓶颈 |
| Day 3 | 因果 mask 与 dropout | 因果 mask（上三角置 -inf）、softmax 后 dropout | 加 mask 与 dropout，对拍 | 能解释 mask 为何用 `-inf` |
| Day 4 | 分块 Attention | 按行分块，避免一次性 N×N；逐块 softmax + rescale | 写分块 attention（块内在线 softmax） | 显存显著下降，理解 rescale |
| Day 5 | 瓶颈分析 | attention 是访存受限；算术强度低；IO 复杂度分析 | 用 roofline 分析 attention | 能说清为什么 attention 慢、怎么优化 |
| Day 6 | 性能对比 | naive vs 分块 vs PyTorch 的 SDPA（`torch.nn.functional.scaled_dot_product_attention`） | 三对比 benchmark | 交出 attention 对比笔记 |
| Day 7 | 月度复习 + 里程碑 | 复盘 M4；P3 打 tag；发《Tensor Core / Attention 实现》博客；提交首个开源 PR | 完成月度自检 + P3 + 博客 + PR | 交出 P3 + 博客 + 首个 PR 提交 |

**M4 月度自检清单：**
- [ ] 能用 WMMA 写 Tensor Core GEMM，对比 FP32 加速比。
- [ ] 会用 CUTLASS 跑 GEMM 并理解 tile 抽象。
- [ ] 手写数值稳定 softmax、online softmax、LayerNorm（含反向）。
- [ ] 手写 naive / 分块 attention，理解 N×N 中间矩阵瓶颈。
- [ ] P3 完成 + 博客发布 + 首个开源 PR 已提交。

## 本月背书行动（在职）

- **发《Tensor Core / CUTLASS / Attention 实现》博客**：附 WMMA vs FP32 加速比、attention 瓶颈分析。
- **提交首个开源 PR**：把 M3 认领的 `good first issue` 完成并提交 PR，进入 review。
- **P3 打 tag + benchmark**。
- 产出口径：**1 篇博客 + 1 个 PR 提交 + P3 项目**。

## Month 4 参考资源

- NVIDIA 博客《Programming Tensor Cores in CUDA 9》、CUDA C++ Programming Guide 的 Warp Matrix 章节
- CUTLASS 官方文档 + `examples/00_basic_gemm`
- 论文《Online Normalizer Calculation for Softmax》
- 《Attention Is All You Need》（attention 公式）、《FlashAttention》论文（M5 精读预热）
