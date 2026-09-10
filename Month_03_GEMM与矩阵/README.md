# Month 3：GEMM 与矩阵运算

> **主题：** 从 naive GEMM 一路优化到 register-tiled GEMM，达到 cuBLAS 同量级；顺带掌握卷积基础与 Roofline 分析。
> **前置：** 完成 M2（会用共享内存、理解合并访问与 occupancy）。
> **月度目标：** 手写 GEMM 达到 cuBLAS 的 ≥50%（理想 ≥80%），能画出性能曲线、用 Roofline 讲清瓶颈；完成 **P2 高性能 GEMM**。

## 月度里程碑

- ✅ Naive GEMM + Roofline 分析，算出算术强度与瓶颈类型。
- ✅ Shared-memory tiling GEMM（分块 + 共享内存缓存）。
- ✅ Register tiling + float4 向量化 + 双缓冲（double buffering）。
- ✅ 理解并实现卷积（im2col / 转 GEMM / 直接卷积）。
- ✅ 完成 P2：与 cuBLAS 对比的性能曲线 + 优化笔记 + 博客。

---

## Week 1：Naive GEMM 与 Roofline

**周目标：** 写出能跑但慢的 naive GEMM，学会用 Roofline 模型定量判断「访存受限 / 计算受限」。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | GEMM 定义 | `C[M,N] = A[M,K] * B[K,N]`，`C[i][j]=Σ_k A[i][k]*B[k][j]`；行主序存储；FLOPs=2·M·N·K | 手写 CPU 三重循环 GEMM + 对拍 | 能算给定规模的 FLOPs 与内存访问量 |
| Day 2 | Naive CUDA GEMM | 每个线程算一个 `C[i][j]`，内层循环 K 累加 | 写 naive CUDA GEMM，与 CPU/cuBLAS 对拍 | 跑通，记录极慢基线（GFLOPs） |
| Day 3 | Roofline 模型 | 算术强度 = FLOPs / bytes；峰值算力 vs 峰值带宽两条线；拐点判断 | 画出 4090 的 Roofline 图，标出 naive GEMM 位置 | 能判断一个 kernel 是访存受限还是计算受限 |
| Day 4 | 分析 naive 为什么慢 | 每个线程读 A/B 都是全局内存、无复用、不合并、重复读 | 用 nsight/带宽计算量化 naive 的访存浪费 | 能说出 naive 的 2–3 个致命问题 |
| Day 5 | cuBLAS 基线 | `cublasSgemm` 用法、`cublasHandle` | 调 cuBLAS 跑同规模 GEMM，记录峰值 GFLOPs 作为目标 | 知道自己要追赶的数字（4090 约 82.6 TFLOPS FP32，Tensor Core FP16 更高） |
| Day 6 | 矩阵分块思想 | 分块（tiling）= 把大矩阵切成 tile，让共享内存缓存一块、复用 | 手工分块 GEMM（block 级），先不优化 | 理解「tile 复用」如何减少全局访存 |
| Day 7 | 复习 + 综合 | naive + 分块雏形 + cuBLAS 三对比，画性能曲线 | 完成 Week 1 benchmark 对比 | 交出「naive vs cuBLAS」基线 + Roofline 笔记 |

**Week 1 自检：** 能算 GEMM 的 FLOPs 与算术强度；能画出并解释 Roofline；能说清 naive GEMM 慢在哪。

---

## Week 2：Shared-Memory Tiling GEMM

**周目标：** 实现经典分块 GEMM：每个 block 把 A、B 的 tile 读进共享内存，复用累加。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 分块 GEMM 结构 | block 负责 `C` 的一个 `BM×BN` tile；循环 K 维，每次读 `BM×BK`、`BK×BN` tile 到共享内存 | 写「单次 K 分块」的 GEMM 骨架 | 跑通 + 对拍 |
| Day 2 | 共享内存加载与同步 | 线程协作把 A/B tile 读进 `__shared__`，`__syncthreads` 分隔读与算 | 完成完整分块 GEMM（K 循环 + 同步） | 性能比 naive 提升数倍，对拍通过 |
| Day 3 | 消除 Bank Conflict | 共享内存 tile 布局加 padding，避免 32 路冲突 | 对比加/不加 padding 的分块 GEMM | 能解释 GEMM 里冲突来自哪、怎么消 |
| Day 4 | Tile 尺寸调优 | `BM/BN/BK` 不同组合影响 occupancy 与复用；测多组 | 扫 BM/BN/BK 组合，画性能热力图 | 能解释「tile 太大占满共享内存 → occupancy 下降」 |
| Day 5 | 双重累加与精度 | K 循环里每线程维护一个 `C` 累加器，注意 FP32 累加精度 | 对比 K 分块内外的累加顺序对精度影响 | 理解累加顺序影响浮点结果 |
| Day 6 | 记录与对比 | 分块 GEMM vs naive vs cuBLAS 的性能差距分析 | 完成 Week 2 benchmark，记录 GFLOPs | 交出「分块 GEMM 优化笔记」 |
| Day 7 | 复习 + 综合 | 复盘分块要点，画出数据流图 | 完善 P2 代码 + 笔记 | 分块 GEMM 稳定通过 + 有性能数字 |

**Week 2 自检：** 能手写分块 GEMM 完整结构；能解释共享内存复用如何减少全局访存；能调 tile 尺寸。

---

## Week 3：Register Tiling、向量化与双缓冲

**周目标：** 在分块基础上做 register tiling + float4 向量化 + 双缓冲，逼近 cuBLAS。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Register Tiling | 每个线程算 `TM×TN` 个小块，存在寄存器里，减少共享内存访问 | 写 register-tiled GEMM（如 4×4），对拍 | 理解「寄存器 tile 进一步减少访存」 |
| Day 2 | 向量化加载 | 用 `float4` 一次读 4 个 float，配合对齐与 tile 尺寸 | 把共享内存加载改 float4 版 | 带宽进一步提升 |
| Day 3 | 双缓冲 | 异步预取下一块 A/B tile，边算边读（`cp.async` 或手动双缓冲） | 写双缓冲版，隐藏访存延迟 | 能解释「计算与访存重叠」 |
| Day 4 | 精读 Boehm 教程 | Simon Boehm《Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》逐步精读 | 复现文中每个优化步骤，记录每次提速 | 理解每个优化各贡献多少 |
| Day 5 | 进一步微调 | 结合 occupancy、寄存器用量、`-use_fast_math`、`__restrict__` | 系统性调优到最佳 GFLOPs | 达到 cuBLAS ≥50% |
| Day 6 | 与 cuBLAS 对标 | 同规模对比自己 GEMM vs cuBLAS，定位残余差距 | 用 nsight 看瓶颈（访存 vs 计算 vs stall） | 能说清「还差在哪、怎么补」 |
| Day 7 | 复习 + 综合 | 完成 P2 benchmark + 性能曲线 | 画「优化步骤 vs GFLOPs」阶梯图 | 交出 P2 高性能 GEMM + 曲线 |

**Week 3 自检：** 能手写 register-tiled GEMM；会用 float4 向量化；能解释双缓冲；能对标 cuBLAS 并定位差距。

---

## Week 4：卷积基础（Convolution）

**周目标：** 理解卷积的几种实现路径（直接卷积 / im2col / 转 GEMM / 隐式 GEMM），并实现一个可用的卷积算子。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 卷积定义与参数 | 卷积 vs 互相关、kernel/stride/padding/dilation、NCHW/NHWC 布局 | 手写 CPU 卷积 + 与 PyTorch `nn.Conv2d` 对拍 | 能算输出尺寸 `(W-F+2P)/S+1` |
| Day 2 | im2col 转 GEMM | 把卷积的每个感受野展开成列，变矩阵乘，复用已有 GEMM | 写 im2col + 调自己 GEMM 实现卷积 | 理解「卷积转 GEMM」的核心 trick |
| Day 3 | 直接卷积 | 每个线程算一个输出点，直接遍历 kernel | 写直接卷积 CUDA kernel | 跑通 + 对拍，理解访存模式 |
| Day 4 | 隐式 GEMM（implicit gemm） | 不显式展开、用索引映射模拟 im2col，省内存（cuDNN 思路） | 写隐式 GEMM 卷积雏形 | 理解 cuDNN 的底层思路 |
| Day 5 | cuDNN 基线 | `cudnnConvolutionForward` 用法，跑同规模对比 | 调 cuDNN 记录性能基线 | 知道工业级卷积的性能目标 |
| Day 6 | 卷积性能分析 | 用 Roofline 分析卷积的算术强度与瓶颈 | 完成卷积各版本 benchmark | 交出卷积对比笔记 |
| Day 7 | 月度复习 + 里程碑 | 复盘 M3；P2 打 tag；发《手写 GEMM 逼近 cuBLAS》博客 | 完成月度自检 + 博客 + P2 README | 交出 P2 GEMM + 卷积 + 博客 |

**M3 月度自检清单：**
- [ ] 能手写 naive → 分块 → register-tiled GEMM，讲清每步优化的原理。
- [ ] 能画出 Roofline 图并判断算子瓶颈。
- [ ] 手写 GEMM 达到 cuBLAS ≥50%。
- [ ] 能实现 im2col / 直接卷积并说明与隐式 GEMM 的关系。
- [ ] P2 完成（性能曲线 + 优化笔记）+ 博客发布。

## 本月背书行动（在职）

- **发《手写 GEMM 逼近 cuBLAS》博客**：附优化步骤阶梯图和 benchmark 数字，这是算子岗最核心的一篇证明。
- **P2 打 tag + benchmark**：仓库附「优化步骤 vs GFLOPs」曲线和与 cuBLAS 的对比表。
- **认领首个开源 issue**：在 PyTorch/Triton 仓库翻 `good first issue`，认领 1 个（先做起来，PR 可以 M4 提交）。
- 产出口径：**1 篇核心博客 + P2 项目 + 1 个 issue 认领**。

## Month 3 参考资源

- PMPP 第 4、6、8 章（tiling、卷积）
- NVIDIA 博客《Matrix Multiplication Background》《How to Optimize GEMM》
- Simon Boehm《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》（精读）
- CUTLASS 文档 GEMM 章节（M4 预热）
- cuDNN Developer Guide（卷积 API 与隐式 GEMM 概念）
