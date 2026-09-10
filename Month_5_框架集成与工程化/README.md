# Month 5：框架集成与工程化

> **主题：** 把算子接入 PyTorch（自定义算子 + Triton），实现 Flash Attention，掌握 Nsight 性能分析。
> **前置：** 完成 M4（有 Tensor Core GEMM、softmax/layernorm/attention 算子）。
> **月度目标：** 算子能「可训练、可反向、可 profiling」；实现一个简化 Flash Attention；完成 **P4 框架集成算子 + profiling 报告**。

## 月度里程碑

- ✅ 用 `torch.autograd.Function` + C++ 扩展封装自定义算子（含反向传播）。
- ✅ 用 Triton 写出 softmax / GEMM 并理解其自动分块与编译。
- ✅ 理解并实现 Flash Attention 的核心思想（分块 + online softmax + 不落地 N×N）。
- ✅ 用 Nsight Compute / Systems 定位并消除一个真实瓶颈。
- ✅ 完成 P4：可训练算子 + profiling 报告 + 博客。

---

## Week 1：PyTorch 自定义算子

**周目标：** 掌握 PyTorch 扩展机制，把 M3/M4 的 CUDA kernel 封装成可训练的自定义算子。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 扩展机制概览 | `torch.utils.cpp_extension`、`load_inline`、setuptools 打包、pybind11 | 跑通官方「用 CUDA 扩展」最小例子 | 环境跑通 |
| Day 2 | `autograd.Function` | `forward`/`backward`、`ctx.save_for_backward`、`grad_output` 链式法则 | 把向量加法包成自定义 Function（含反向） | 理解反向梯度如何写 |
| Day 3 | 封装一个算子 | 把 M4 的 LayerNorm（前向+反向）包成自定义算子 | 写 `LayerNormFunction`，接入 PyTorch | 能训练、`gradcheck` 通过 |
| Day 4 | `gradcheck` 与对拍 | `torch.autograd.gradcheck` 数值验梯度，与 PyTorch 参考算子对拍 | 对自己的算子跑 gradcheck + 精度对拍 | 梯度正确性得到验证 |
| Day 5 | 封装 GEMM/Softmax | 把 GEMM、softmax 也接成自定义算子，支持 dtype/流 | 完成 2–3 个算子封装 | 多个算子可训练 |
| Day 6 | 性能集成 | 在真实模型（一个小 MLP/Transformer 层）里替换算子，测端到端 | 用自定义算子拼一个前向层，对比 PyTorch 原生 | 能展示自定义算子的端到端价值 |
| Day 7 | 复习 + 综合 | 整理封装模板（forward/backward/pybind11/setup.py） | 完成 Week 1 笔记 + 可复用模板 | 交出「PyTorch 自定义算子模板」 |

**Week 1 自检：** 能把一个 CUDA kernel 封装成带反向的 PyTorch 自定义算子；gradcheck 通过；模型里能训练。

---

## Week 2：Triton

**周目标：** 用 Triton 快速写算子，理解其自动分块、编译与 JIT 机制，作为 CUDA 的高效补充。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Triton 概览 | `@triton.jit`、`program_id`、block 编程模型、自动 tile | 跑官方 vector add / softmax 示例 | 环境跑通 |
| Day 2 | Triton Softmax | Triton 的 block 级 softmax（load → 归约 → store），自动处理跨 block | 写 Triton softmax，对拍 | 跑通，理解 block 编程 |
| Day 3 | Triton GEMM | Triton 矩阵乘（自动 tile + 累加），对比自写 CUDA | 写 Triton GEMM，对拍 + 计时 | 理解 Triton 如何抽象 tiling |
| Day 4 | Triton vs CUDA | 同一算子的 Triton 与手写 CUDA 性能对比，理解适用场景 | softmax/GEMM 双对比 | 能说清「何时手写 CUDA、何时用 Triton」 |
| Day 5 | Triton 编译机制 | Triton → Triton IR → LLVM IR → PTX 的编译链；`TRITON_CACHE_DIR` | 用 `triton.compile` 或日志看中间 IR | 理解 JIT 与缓存 |
| Day 6 | 高级特性 | `tl.load` mask、`tl.dot`（自动用 Tensor Core）、`tl.atomic_*` | 写一个带 mask 和 `tl.dot` 的算子 | 会用 Triton 关键原语 |
| Day 7 | 复习 + 综合 | 整理 Triton 笔记，把 Triton softmax 也接成 PyTorch 算子 | 完成 Week 2 笔记 | 交出 Triton 笔记 + 算子 |

**Week 2 自检：** 能用 Triton 写 softmax/GEMM；能解释 Triton 的 block 模型与自动分块；能对比 Triton 与手写 CUDA。

---

## Week 3：Flash Attention

**周目标：** 精读并实现 Flash Attention 核心思想：分块 + online softmax + 避免 N×N 落地。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 论文精读 | FlashAttention 1：分块计算、online softmax rescale、IO 复杂度 O(N²d²/M) | 通读论文，画出分块计算图 | 能讲清「为什么不把 S=N×N 写回显存」 |
| Day 2 | 分块 online softmax 复习 | 把 M4 的分块 attention 与 online softmax 串起来，理解 rescale | 手写「两块的 online softmax 合并」伪代码 | 理解跨块 rescale 公式 |
| Day 3 | Triton 实现 Flash Attention | 用 Triton 写 flash attention（官方 tutorial 简化版） | 复现 Triton FA tutorial，对拍 | 跑通简化 FA |
| Day 4 | 关键细节 | causal mask、`tl.dot` 用 Tensor Core、`l_i`/`m_i` 状态维护 | 加 causal mask，处理反向 | 简化 FA 支持 mask + backward |
| Day 5 | 与 PyTorch SDPA 对比 | `torch.nn.functional.scaled_dot_product_attention` 已是融合实现 | FA vs SDPA vs naive 三对比 benchmark | 看到显存与速度优势 |
| Day 6 | 读 Triton FA 源码 | 精读 Triton 官方 `03-matrix-multiplication` 或 FA 内核源码 | 逐行读，记数据流 | 能讲清 FA 内核每一段 |
| Day 7 | 复习 + 综合 | 完成 FA 笔记 + benchmark；发《Flash Attention 实现笔记》博客 | 月度 FA 成果 + 博客 | 交出 FA 实现 + 博客 |

**Week 3 自检：** 能讲清 Flash Attention 的 IO 优化核心；能用 Triton 复现简化 FA；能对比 FA/naive 的显存与速度。

---

## Week 4：Nsight 性能分析

**周目标：** 掌握 Nsight Compute / Systems，能定位并消除真实 kernel 的瓶颈。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Nsight 工具链 | `ncu`（kernel 级）、`nsys`（系统级）、`nsight-sys profile`；AutoDL 容器内可能需 `--privileged` 权限 | 搭建能跑 ncu 的环境（4090 上测，或用本地/可 root 镜像） | 环境可 profiling |
| Day 2 | ncu 核心指标 | `sm__throughput`、`gpu__time_duration`、内存吞吐、warp stall 原因、occupancy | 对自己的 GEMM 跑 `ncu --set full` | 能读出一份 ncu 报告 |
| Day 3 | 定位瓶颈 | 通过指标判断：访存受限 vs 计算受限 vs 延迟/stall | 分析 M3 GEMM 的 ncu 报告，找出 top stall 原因 | 能说清瓶颈类型 + 证据 |
| Day 4 | 优化闭环 | 根据 ncu 提示改代码（tile/向量化/occupancy），再测 | 优化一个算子，ncu 前后对比 | 完成「profile → 优化 → 复测」闭环 |
| Day 5 | nsys 时间线 | `nsys profile` 看 kernel 调用序列、拷贝与计算的 overlap | 跑一个多 kernel 程序，读 nsys 时间线 | 能定位「拷贝没 overlap」等问题 |
| Day 6 | 写 profiling 报告 | 把一次完整的优化过程写成报告（瓶颈 → 手段 → 数字） | 完成 P4 profiling 报告 | 交出可放进 portfolio 的报告 |
| Day 7 | 月度复习 + 里程碑 | 复盘 M5；P4 打 tag；主攻开源贡献（vLLM/xFormers 算子） | 月度自检 + P4 + 开源 PR | 交出 P4 + profiling 报告 + 1 个 PR |

**M5 月度自检清单：**
- [ ] 能把 CUDA kernel 封装成带反向的 PyTorch 自定义算子（gradcheck 通过）。
- [ ] 能用 Triton 写 softmax/GEMM，能对比 Triton vs 手写 CUDA。
- [ ] 能实现简化 Flash Attention 并讲清其 IO 优化。
- [ ] 能用 ncu 定位并消除一个真实瓶颈，写出 profiling 报告。
- [ ] P4 完成 + 博客 + 1 个开源 PR。

## 本月背书行动（在职）

- **发《Flash Attention 实现笔记》博客**（在职者展示前沿理解的核心证明）。
- **主攻开源贡献**：在 vLLM / xFormers / PyTorch 里找一个算子相关的 `good first issue` 或性能小优化，提交 PR。
- **P4 打 tag + profiling 报告**作为作品集。
- 产出口径：**1 篇博客 + 1 个 PR + P4（含 profiling 报告）**。

## Month 5 参考资源

- PyTorch 官方《Custom C++ and CUDA Extensions》教程
- Triton 官方 tutorials（softmax、matrix-multiplication、fused attention）
- FlashAttention 论文 + 作者博客（Elvis 的 blog）
- NVIDIA Nsight Compute / Systems 官方文档
