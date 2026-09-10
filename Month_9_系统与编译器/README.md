# Month 9：系统与编译器

> **主题：** 下沉到 PTX/SASS 与编译器层，系统理解数值精度（BF16/FP8/INT8），了解分布式算子（NCCL/all-reduce）。
> **前置：** 完成 M8（能读 CUTLASS 源码、有性能优化 PR）。
> **月度目标：** 能读 PTX/SASS 并解释优化；理解数值精度对算子的影响；理解分布式通信算子；发布 1 篇系统/数值博客；累计 3–5 个 merged PR。

## 月度里程碑

- ✅ 能读 PTX/SASS，用 `nvdisasm`/`cuobjdump` 看自己 kernel 的指令。
- ✅ 理解 nvcc 的优化选项与编译器行为（内联、展开、fast-math）。
- ✅ 深入 BF16/FP8/INT8 的数值行为，理解 FP8 缩放与误差。
- ✅ 理解 Triton 编译链与 NCCL all-reduce 的 ring/tree 算法、通信与计算重叠。
- ✅ 累计 3–5 个 merged PR。

---

## Week 1：PTX / SASS

**周目标：** 从「写 CUDA」下沉到「读编译产物」，能对着 SASS 解释性能。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 编译链概览 | CUDA C++ → PTX（虚拟 ISA）→ SASS（真实机器码），`nvcc -ptx`、`-cubin` | 对自己 kernel 生成 `.ptx`、`.cubin` | 理解两级 ISA |
| Day 2 | PTX 基础 | 寄存器、`ld.global`/`st.shared`、`mad`/`fma`、`bar.sync` | 读自己 GEMM 的 PTX | 能认出关键指令 |
| Day 3 | SASS 反汇编 | `nvdisasm`/`cuobjdump -sass`，看真实指令调度 | 反汇编一个 kernel，看指令序列 | 能对着 SASS 说性能 |
| Day 4 | 指令级分析 | FFMA 吞吐、访存指令数、寄存器重命名、双发射 | 数一个循环里的 FFMA 与访存指令 | 能估算指令级瓶颈 |
| Day 5 | 从 SASS 看优化 | 为什么 tiling/向量化在 SASS 层体现为更少访存指令 | 对比 naive 与 tiled 的 SASS | 把优化与指令对上号 |
| Day 6 | 编译器优化旗标 | `-use_fast_math`、`--fmad`、`--maxrregcount`、`-lineinfo` | 对比不同旗标的 SASS 差异 | 理解旗标如何改代码 |
| Day 7 | 复习 + 综合 | 整理 PTX/SASS 笔记 | 完成 Week 1 笔记 | 交出 PTX/SASS 笔记 |

**Week 1 自检：** 能生成并读 PTX/SASS；能解释 tiling/向量化在指令层的体现；会用 nvdisasm。

---

## Week 2：nvcc 优化与编译器

**周目标：** 理解 nvcc 编译过程与优化，掌握控制代码生成的手段。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | nvcc 编译流程 | `cudafe`、`cicc`、`ptxas` 各阶段；`--dryrun` 看全流程 | 用 `nvcc --dryrun` 看编译步骤 | 理解编译各阶段 |
| Day 2 | ptxas 优化 | `ptxas` 做寄存器分配、指令调度、`-v` 看资源 | 对比 `-O0/-O3` 的 ptxas 输出 | 理解 ptxas 作用 |
| Day 3 | 内联与展开 | `__forceinline__`、`#pragma unroll`、模板如何影响代码生成 | 测内联/展开对性能的影响 | 理解代码生成控制 |
| Day 4 | fast-math 与精度 | `-use_fast_math` 的 `__sinf/__expf/__powf`、`__fmul_rn` 舍入 | 对比 fast-math 前后结果与速度 | 能说清 fast-math 的精度代价 |
| Day 5 | 约束与断言 | `__restrict__`、`__launch_bounds__`、`__builtin_assume` | 用 `__restrict__`/`__launch_bounds__` 优化 | 理解给编译器更多信息 |
| Day 6 | 编译器与架构 | `-arch` 如何影响可用指令（sm_89 / sm_80 / sm_90 差异） | 同 kernel 编三个 arch 对比 | 理解 arch 决定的指令集 |
| Day 7 | 复习 + 综合 | 整理 nvcc/编译器笔记 | 完成 Week 2 笔记 | 交出编译器笔记 |

**Week 2 自检：** 理解 nvcc/ptxas 流程；会控制内联/展开/fast-math；会用 `__restrict__`/`__launch_bounds__`。

---

## Week 3：数值精度与 Triton 编译

**周目标：** 深入 BF16/FP8/INT8 数值行为，理解 Triton 的编译链。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | FP16/BF16 深入 | 半精度的舍入、上溢/下溢、BF16 的 8 位尾数 | 写精度实验：加/乘的误差 | 能说清 BF16 为何「范围大精度低」 |
| Day 2 | FP8 深入 | E4M3/E5M2、缩放因子、误差放大、FP8 训练/推理 | 实现 FP8 缩放 matmul，测误差 | 理解 FP8 缩放机制 |
| Day 3 | 混合精度训练 | master weight、loss scaling、梯度缩放 | 读混合精度训练资料 | 理解 loss scaling 防下溢 |
| Day 4 | Triton 编译链 | Triton → MLIR/Triton IR → LLVM IR → PTX → SASS | 用 `triton.compile` 看各阶段 IR | 理解 Triton 编译流程 |
| Day 5 | Triton 代码生成 | 自动 tiling/向量化/Tensor Core 映射如何体现在 IR | 对比 Triton 与手写 CUDA 的 SASS | 理解 Triton 优化能力 |
| Day 6 | 数值调试 | NaN/Inf 定位、`--fmad=false` 定位舍入 bug | 复现并定位一个精度 bug | 有定位数值 bug 的方法 |
| Day 7 | 复习 + 综合 | 发《数值精度》博客 | 完成 Week 3 笔记 + 博客 | 交出数值精度笔记 + 博客 |

**Week 3 自检：** 能说清 BF16/FP8/INT8 的表示与误差；理解 FP8 缩放与混合精度训练；能看 Triton 编译 IR。

---

## Week 4：分布式算子（NCCL / all-reduce）

**周目标：** 理解多 GPU 通信算子与「通信/计算重叠」，了解 NCCL 的 ring/tree 算法。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 分布式基础 | 数据并行/模型并行/流水线并行、all-reduce/all-gather 原语 | 读 PyTorch DDP 概念 | 理解各并行策略 |
| Day 2 | all-reduce 算法 | ring all-reduce、tree all-reduce、带宽 vs 延迟权衡 | 画 ring 与 tree 的数据流图 | 能讲清 ring all-reduce |
| Day 3 | NCCL 概览 | NCCL 是 NVIDIA 的集合通信库，看其 API 与算法选择 | 读 NCCL 文档 | 理解 NCCL 定位 |
| Day 4 | 通信与计算重叠 | 反向里梯度 all-reduce 与下一层计算重叠（`bucket`） | 读 DDP/ZeRO 的重叠思路 | 能说清 overlap 收益 |
| Day 5 | 通信算子性能 | 带宽、延迟、`nccl-tests` 基准 | 跑 `nccl-tests`（如有多卡）或读结果 | 理解通信 benchmark |
| Day 6 | 算子在分布式中的角色 | GEMM/attention 在张量并行里如何切分、通信如何插入 | 读 Megatron 的 TP 切分 | 理解算子与通信的结合 |
| Day 7 | 月度复习 | 复盘 M9；月度自检；累计 PR 检查 | 完成自检 + 归档 | 交出 M9 成果（累计 3–5 PR） |

**M9 月度自检清单：**
- [ ] 能读 PTX/SASS 并解释 tiling/向量化的指令层体现。
- [ ] 理解 nvcc/ptxas 流程与 fast-math/`__restrict__`/`__launch_bounds__`。
- [ ] 能说清 BF16/FP8/INT8 数值行为与 FP8 缩放、loss scaling。
- [ ] 理解 Triton 编译链、NCCL ring/tree all-reduce、通信与计算重叠。
- [ ] 累计 3–5 个 merged PR + 1 篇系统/数值博客。

## 本月背书行动（在职）

- **发《数值精度 / 系统》博客**：展示底层与编译器功底。
- **累计 3–5 个 merged PR**：盘点并补齐开源贡献。
- 产出口径：**1 篇博客 + 累计 3–5 PR**。

## Month 9 参考资源

- CUDA C++ Programming Guide：PTX、nvcc 章节；`nvdisasm`/`cuobjdump` 文档
- NVIDIA FP8 白皮书、混合精度训练（Apex/amp）资料
- Triton 编译器文档
- NCCL 官方文档、Megatron-LM 张量并行源码
