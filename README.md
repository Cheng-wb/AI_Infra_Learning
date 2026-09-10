# AI Infra 算子开发工程师 · 12 个月学习计划（6 个月主线 + 6 个月冲刺）

> **目标岗位：** NVIDIA 算子开发工程师（CUDA Kernel / Operator Development）
> **阶段划分：**
> - **第 6 个月 checkpoint：** 具备「可投第一份 AI Infra / 算子岗」的能力与作品集。
> - **第 12 个月最终：** 对标 NVIDIA 正式算子岗，能独立分析算子瓶颈、写出「正确 + 高性能 + 可接入 PyTorch」的 CUDA 算子，并用 Roofline / Nsight 讲清「为什么快 / 为什么慢」。
> **学习节奏：** 每天 4 小时；**运行环境：** AutoDL 租用 RTX 4090（Ada / sm_89，24GB），按需租 A100/H100 测 Hopper 特性。
> **当前身份：** 在职 → 背书主攻「**开源贡献 + 技术博客 + 项目作品集 + 社招内推**」，不依赖实习 / 学生竞赛。

---

## 一、这个岗位到底要什么（JD 拆解）

NVIDIA 算子开发工程师（算子方向）日常工作本质是：**把深度学习框架里的一个算子（GEMM、卷积、Softmax、LayerNorm、Attention、量化、通信…），写成在目标 GPU 上又快又正确的 CUDA Kernel，并接入 PyTorch/TensorFlow，用 Nsight 验证性能。**

把它拆成可学习、可考核的能力项：

| # | 能力项 | 对应月份 | 验收方式 |
| --- | --- | --- | --- |
| 1 | C++ 编程（指针/模板/STL/内存/编译） | M1 | 用 C++ 独立实现数据结构 + 单测 |
| 2 | CUDA 编程模型（线程/Block/Grid/内存层次） | M1 | 手写并验证向量加法等基础 kernel |
| 3 | GPU 架构（SM/Warp/Tensor Core/带宽） | M1–M4 | 能画出内存层次图、解释 warp 调度 |
| 4 | 内存优化（合并访问/Bank Conflict/共享内存/Tiling） | M2–M3 | 归约/转置/GEMM 达到性能目标 |
| 5 | 并行原语（Reduction/Scan/Transpose） | M2 | 三件套 kernel 通过 CPU 对拍 |
| 6 | GEMM 与矩阵运算 | M3 | tiled GEMM 达到 cuBLAS 同量级 |
| 7 | Tensor Core 编程 + CUTLASS | M4 | WMMA/CUTLASS GEMM 跑通并提速 |
| 8 | 框架集成（PyTorch 自定义算子/Triton） | M5 | 算子可训练、可反向传播 |
| 9 | 性能分析（Roofline/Nsight Compute/Systems） | M3–M6 | 能定位并消除一个真实瓶颈 |
| 10 | 现代架构（Hopper TMA/wgmma/CUDA Graph） | M7 | 概念 + 代码结构 + A100/H100 实测 |
| 11 | 系统与编译器（PTX/SASS/nvcc/数值精度） | M9 | 能读 SASS、解释精度差异 |
| 12 | 分布式算子（NCCL/all-reduce/通信隐藏） | M9 | 理解通信算子与计算重叠 |
| 13 | 开源工程能力（贡献 PR/读大型代码库） | M6–M11 | 累计 3–5 个 merged PR |
| 14 | 面试能力（八股/刷题/算子案例/系统设计） | M10–M11 | 模拟面试通过、拿到 offer |

> **一句话定位：** 这不是「会写 Python/调 API」，而是「会写贴近硬件的 CUDA C++，并理解每一行代码在 GPU 上如何执行」。前 6 个月打地基 + 出作品，后 6 个月冲高度 + 出背书。

---

## 二、学习路线图

```text
【Phase 1 · M1–M6 · 主线：从零到「能写高性能算子 + 有作品集」】
Month 1  C++ 与 CUDA 编程基础        → 能写/编译/验证第一个 CUDA kernel
Month 2  内存优化与核心数据算子       → Reduction / Scan / Transpose 三件套
Month 3  GEMM 与矩阵运算             → naive → tiled → register-tiled GEMM
Month 4  Tensor Core 与进阶算子       → WMMA / CUTLASS / Softmax / Attention
Month 5  框架集成与工程化             → PyTorch 自定义算子 / Triton / Flash Attention / Nsight
Month 6  进阶整合 + 量化入门 + 首个 PR → Capstone v1 完成，开始开源贡献
                    ───────────────────────────────
                    checkpoint：可投第一份 AI Infra / 算子岗
                    ───────────────────────────────
【Phase 2 · M7–M12 · 冲刺：深挖 + 背书 + 面试】
Month 7  现代架构与高级特性           → Hopper TMA/wgmma / CUDA Graph / async copy
Month 8  开源深挖                     → CUTLASS 源码 / vLLM、xFormers 贡献 / FlashAttention 复现
Month 9  系统与编译器                 → PTX/SASS / nvcc / Triton 编译 / 数值精度 / NCCL
Month 10 面试准备                     → 八股 / 刷题 / 算子案例 / 系统设计 / 简历 + 内推
Month 11 投递与面试实战               → 海投 / 面试 / 复盘 / 持续开源
Month 12 收尾与决策                   → 深度项目收口 / offer 决策
```

---

## 三、贯穿 12 个月的 7 个项目（求职作品集）

每个项目都是「能放进简历 + 能讲清楚」的完整闭环，均包含：CPU/朴素实现对拍 → CUDA 实现 → 性能 benchmark → 文档。

| 项目 | 周期 | 内容 | 产出 |
| --- | --- | --- | --- |
| **P1 基础算子库** | M1–M2 | element-wise / reduction / transpose / scan + 统一校验框架 | 可复用算子库 + 单测 |
| **P2 高性能 GEMM** | M3 | naive → shared-mem tiling → register tiling + 向量化 | 达到 cuBLAS 同量级，附性能曲线 |
| **P3 Tensor Core 算子集** | M4 | WMMA GEMM + CUTLASS + Softmax/LayerNorm/Attention | 对比普通 GEMM 的加速比 |
| **P4 框架集成算子** | M5 | PyTorch 自定义算子 + Triton + Flash Attention + Nsight 报告 | 可训练算子 + profiling 报告 |
| **P5 Capstone v1 算子库** | M6 | 融合算子（fused LN+GEMM / 简化 attention）+ 量化入门 + benchmark | 完整开源库 + README |
| **P6 明星项目** | M7–M9 | Flash Attention 2/3 复现或深度优化 + Hopper 特性 + 分布式算子 | 有分量的开源项目 + 博客 |
| **P7 开源贡献成果** | M6–M11 | PyTorch / Triton / CUTLASS / vLLM / xFormers 的 PR | 3–5 个 merged PR |

> 核心原则：**不要只学算法，持续完成「理论 → 建模 → 编码 → 实验 → benchmark」的闭环。** 每个算子都经历：CPU 参考实现（定正确性）→ GPU 实现 → 数值对拍 → 计时 → 优化 → 画性能曲线。

---

## 四、硬件与环境（AutoDL · RTX 4090）

**你在 AutoDL 租用 RTX 4090（24GB）。** 这是比 Colab T4 好得多的选择：算力约 10 倍、显存更大、数据盘持久（关机不丢环境/代码）、SSH + JupyterLab 自由、镜像自带 CUDA + PyTorch。关键参数与边界：

| 环境 | GPU | 计算能力 sm | 能跑什么 | 不能跑什么 |
| --- | --- | --- | --- | --- |
| **AutoDL 4090** | RTX 4090 | sm_89（Ada） | 全部基础 CUDA、shared memory、WMMA、`cp.async`、FP8（消费级卡吞吐打折）、强 FP16/BF16/INT8 Tensor Core | 无 TMA / wgmma / Thread Block Cluster / DSMEM（Hopper sm_90 专属） |
| 云 A100（按需租） | A100 | sm_80（Ampere） | 4090 全部 + 更强 Tensor Core、更高显存带宽 | Hopper 特性仍不行 |
| 云 H100（按需租） | H100 | sm_90（Hopper） | 全部，含 TMA / wgmma / Thread Block Cluster | — |

**编译 arch：** 4090 用 `-arch=sm_89`（或 `compute_89`）；用 `#if __CUDA_ARCH__ >= 900` 做 Hopper 特性条件编译，保证同一份代码在 4090 能跑、在 H100 走新路径。

**4090 关键数字（Roofline 会用）：** 峰值 FP32 ≈ 82.6 TFLOPS，Tensor Core FP16 ≈ 165 TFLOPS，显存带宽 ≈ 1008 GB/s（对比 T4 的 8.1 TFLOPS / 320 GB/s）。

**策略：**
- **M1–M6 全部实验在 4090 上完成**（4090 有 Tensor Core、支持 WMMA 和 `cp.async`，够学会 Tensor Core 编程与异步拷贝）。
- **M7 的 Hopper 特性（TMA/wgmma/cluster）** 属于「看文档 + 读源码 + 概念理解」，**仅按需租 1–2 小时 H100 做实测**，4090 不强求——但概念和代码结构一定要看懂，面试会问。
- AutoDL 关机释放算力但**数据盘保留**；务必把代码 push 到 GitHub，避免只依赖单块数据盘。
- 4090 无 NVLink，M9 的 NCCL 多卡 all-reduce 实测需多卡实例或只读概念，单卡学习够用。

**工具链：** `nvcc`（AutoDL 镜像已装）、`g++`、`CMake`、`pybind11` / `torch.utils.cpp_extension`（M5）、`triton`（M5）、`cuda-python`/`cupy`（可选）；Nsight Compute / Systems 用 `ncu`/`nsys`（M5–M6，容器内可能需 `--privileged` 权限，若无则用本地或可 root 的镜像）。

---

## 五、每天 4 小时怎么分配

| 时间段 | 时长 | 内容 |
| --- | --- | --- |
| 理论 / 精读 | 40 min | 读当天知识点、官方文档、源码片段，记笔记 |
| 动手编码 | 2 h | 写当天 kernel / 练习，跑通、对拍 |
| 实验 / 对比 | 40 min | 改参数、计时、画性能曲线、与参考实现对比 |
| 总结复盘 | 40 min | 写当日笔记（结论 + 踩坑 + 性能数字），完成自检 |

> 每周第 7 天为「复习 + 综合动手 + 周笔记」；Phase 2 的第 7 天逐步转为「开源贡献 + 面试 + 简历」时间。

---

## 六、十二个月月度概览

**Phase 1（M1–M6）主线：**

| 月 | 主题 | 里程碑产出 |
| --- | --- | --- |
| M1 | C++ 与 CUDA 基础 | 第一个验证过的 kernel |
| M2 | 内存优化与核心算子 | P1 基础算子库 |
| M3 | GEMM 与矩阵 | P2 高性能 GEMM |
| M4 | Tensor Core + 进阶算子 | P3 Tensor Core 算子集 |
| M5 | 框架集成 + 工程化 | P4 可训练算子 + profiling |
| M6 | 进阶整合 + 量化 + 首个 PR | P5 Capstone v1 + 开源起航 |

**Phase 2（M7–M12）冲刺：**

| 月 | 主题 | 里程碑产出 |
| --- | --- | --- |
| M7 | 现代架构与高级特性 | Hopper 特性实验 + 博客 |
| M8 | 开源深挖 | 有分量的 PR / FlashAttention 复现 |
| M9 | 系统与编译器 | PTX/SASS 笔记 + 数值精度 |
| M10 | 面试准备 | 简历 + 内推启动 + 刷题 |
| M11 | 投递与面试实战 | 面试复盘 + 持续开源 |
| M12 | 收尾与决策 | offer 决策 / 深度项目收口 |

---

## 七、成果背书与求职路径（简历怎么过）

> **核心认知：** 你能进面试，靠的不是「我学过」，而是「我做出过、且能被第三方验证」。背书和理论学习**并行**，从第 1 个月就开始。你是**在职**，实习/学生竞赛不适用，所以主攻下面 L1/L2/L3/L5/L6。

背书的含金量从低到高排序：

**L1 — GitHub 作品集项目（M1 起持续，人人可做）**
- 把每个里程碑做成带 README、单测、benchmark 数字、性能曲线图的公开仓库。
- 关键：**数字可复现**（给出环境、`nvidia-smi`、对比 cuBLAS/cuDNN/PyTorch 的加速比/带宽占比）。
- 一个「我手写的 GEMM 达到 cuBLAS 88%」的 benchmark 截图，比 10 个没说清楚的 demo 值钱。

**L2 — 技术博客（M2 起，沉淀个人品牌）**
- 每完成一个里程碑写一篇深度文，讲清「原理 → 实现 → 踩坑 → 性能数字」。
- 发在知乎 / 公众号 / 个人博客 / Medium，标题如《手写 CUDA 归约，从 40GB/s 到 300GB/s》《手写 GEMM 逼近 cuBLAS》《Flash Attention 实现笔记》。
- 面试时甩链接，比口头自述可信得多。

**L3 — 开源贡献（M3 起，含金量陡增，在职者最强背书）**
- 目标仓库：**PyTorch、Triton、CUTLASS、vLLM、xFormers、Flash-Attention**。
- 从 `good first issue` / 文档 / 单测 / 小 bug / 性能小优化入手，目标是 **累计 3–5 个 merged PR**，其中至少 1 个是「有分量的性能优化」。
- 一个 merged PR（哪怕修一个边界 bug 或补一个算子测试）在算子岗简历里是硬通货，远胜一堆自练 demo。

**L4 — 竞赛（你已工作，基本不适用）**
- ASC/ISC/SC 学生超算竞赛等需在校生身份，跳过；但可了解 MLPerf 的 benchmark 方法论，面试谈资加分。

**L5 — 认证（可选，快速补一块）**
- NVIDIA Deep Learning Institute（DLI）证书：CUDA C/C++、加速计算课程，花 1–2 天可拿，简历「有官方背书」的速效项。

**L6 — 内推 / 社招（最强，M10 启动）**
- 在职社招靠**内推 + 作品集**：找目标公司（NVIDIA、字节/阿里/腾讯 AI infra、地平线、燧原、摩尔线程、天数智芯等）的内部员工内推。
- 有 M1–M9 的作品集 + 开源 PR 再去投，命中率完全不同。

**各月背书行动（与月计划同步推进）：**

| 月 | 本月背书行动 | 目标产出 |
| --- | --- | --- |
| M1 | 建 GitHub 仓库、规范 commit、写好 README；报 1 门 DLI 课程 | 公开仓库 + 首个可复现 demo |
| M2 | 发第 1 篇博客（归约/转置优化）；P1 打 tag + benchmark | 1 篇博客 + P1 数字 |
| M3 | 发《手写 GEMM》博客；认领 PyTorch/Triton `good first issue` | 1 篇博客 + 首个 issue 认领 |
| M4 | 首个开源 PR 提交；《Tensor Core/CUTLASS》博客 | 1 个 PR 提交 |
| M5 | 主攻开源（vLLM/xFormers/PyTorch 算子）；《Flash Attention》博客 | 1 个 PR + 1 篇博客 |
| M6 | 争取 1–2 个 merged PR；Capstone v1 README + benchmark | 1–2 merged PR + P5 |
| M7 | 《Hopper TMA/wgmma 实测》博客；持续开源 | 1 篇博客 |
| M8 | 1 个有分量的性能优化 PR；开源深挖笔记 | 性能 PR |
| M9 | 系统/数值精度博客；累计 3–5 merged PR | 累计 3–5 PR |
| M10 | 打磨简历 + 内推启动（NVIDIA 及 AI infra 公司）+ 刷题 | 简历 + 内推渠道 |
| M11 | 海投 20+ 家、面试复盘、持续开源 | 面试记录 + 复盘 |
| M12 | offer 决策 / 深化明星项目 | offer 或明确下一步 |

> **一句话：** 学习计划解决「会不会」，背书行动解决「别人信不信」。背书决定你 12 个月后能不能拿到面试和 offer。

---

## 八、资源清单

**CUDA 官方（必读，一手资料）：**
- 《CUDA C++ Programming Guide》（编程模型、内存、异步、特性，M1–M12 常查）
- 《CUDA C++ Best Practices Guide》（合并访问、occupancy、优化清单，M2–M3 精读）
- NVIDIA Nsight Compute 文档 / Kernel Profiling Guide（M5–M6）
- CUTLASS 官方文档与 `examples/`（M4、M8）

**书籍：**
- 《Programming Massively Parallel Processors》(PMPP, Hwu et al.) —— 算子开发核心教材，M1–M4 主线
- 《CUDA C 编程权威指南》（入门可快速过）

**博客/教程：**
- NVIDIA 开发者博客：`How to Optimize GEMM`、`Matrix Multiplication Background`、`Optimizing Parallel Reduction`、`An Even Easier Introduction to CUDA`
- Simon Boehm《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》（M3 精读）
- Lei Mao、Simon Willison、Horace He 的 Triton/算子博客（M5）

**论文（M4–M9 读，理解 idea 而非背）：**
- Flash Attention 1/2/3（M5、M8）
- Online Softmax（M4）
- CUTLASS 论文《CUTLASS: Fast Linear Algebra in CUDA C++》（M4、M8）
- GPTQ / FP8-LLM / NVIDIA FP8（M6、M9）

---

## 九、最终验收标准

**第 6 个月 checkpoint 必须能独立做到：**
1. 不看模板，写出正确、可编译、带错误检查的 CUDA kernel，并用 CPU 参考实现对拍验证。
2. 给定一个算子，说出其算术强度、用 Roofline 判断「计算受限/访存受限」，给出 ≥2 种优化手段。
3. 自己写的 GEMM 达到 cuBLAS 同量级（≥50%，理想 ≥80%）。
4. 把 CUDA kernel 封装成 PyTorch 自定义算子（含反向），模型里可正常训练。
5. 用 Nsight Compute 定位一个真实瓶颈并消除它。

**第 12 个月最终必须能独立做到：**
6. 讲清 Hopper 新特性（TMA/wgmma/cluster）的动机与代码结构，并做过实测。
7. 读懂一段 PTX/SASS，解释数值精度（FP32/BF16/FP8/INT8）对算子结果的影响。
8. 拥有 3–5 个 merged 开源 PR、一个明星项目、8+ 篇博客。
9. 像面试一样讲清任意算子的「问题定义 → 并行分解 → 内存访问 → 优化手段 → 性能结果」。
10. 拿到 NVIDIA 或头部 AI infra 公司的 offer，或进入终面。

**一句话总结：** 12 个月后，你要从「会调 PyTorch API」变成「能写 CUDA Kernel、并有理有据地把它做到快的人」，且这些能力都有公开、可验证的成果背书。
