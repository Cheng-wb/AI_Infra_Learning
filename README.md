# 2026 AI Kernel / AI Systems Engineer · 12 个月学习计划

> **版本日期：2026-09-11**
> **目标岗位：** AI Kernel Engineer / GPU Performance Engineer / AI Systems Engineer / Inference Systems Engineer
> **建议投入：** 每天约 4 小时，每周 7 天中 Day 7 以 benchmark、复盘、文档、PR 为主
> **主环境：** NVIDIA RTX 4090（Ada / sm_89）可覆盖大部分 CUDA、Triton、cuTile 基础实验；按需租 H100/H200/B200/GB200 做 Hopper/Blackwell 与多卡实验
> **学习原则：** 不以「看完」为完成标准，只认 **correctness + benchmark + profiler evidence + reproducibility + code review**。

---

## 仓库结构

本仓库同时承载「学习计划」与「代码项目」两部分：

- `Month_01` … `Month_12`：**学习计划**。每个月份文件夹内是本月的逐周、逐日任务表与验收标准（详见各 `README.md`）。
- `kernels/` `operators/` `compiler_lab/` `runtime/` `distributed/` `benchmarks/` `profiling/` `tests/` `docs/`：**代码项目结构**（见下方第 5 节），所有实验代码、benchmark、文档按此落盘。

---

## 0. 这份路线和传统 CUDA 路线有什么不同

这不是单纯的 CUDA Kernel 学习表，而是按 2026 年 AI Infra 的真实工作栈组织：

1. **Kernel：** CUDA C++、内存层次、并行原语、GEMM、Tensor Core、Attention、量化、MoE。
2. **Kernel DSL：** Triton、CuTe/CUTLASS DSL、cuTile Python。
3. **Framework / Compiler：** PyTorch custom op、`torch.library`、`torch.compile`、Inductor、IR、PTX/SASS、基础 MLIR/LLVM 思维。
4. **Inference Runtime：** KV Cache、PagedAttention、continuous batching、chunked prefill、prefix caching、CUDA Graph、sampling、speculative decoding。
5. **Distributed：** NCCL、ReduceScatter/AllGather/AllToAll、TP/DP/PP/EP/CP、NVLink/NVSwitch、InfiniBand/RoCE、NVSHMEM。
6. **Production：** TTFT/TPOT/throughput、调度、可观测性、容量规划、故障与 OOM、性能回归。
7. **工程背书：** benchmark、profiling report、技术博客、开源 PR、明星项目、面试与投递。

### 12 个月结束时应达到的能力

- 能从算子数学定义推导并行分解、数据布局、算术强度和瓶颈。
- 能在 CUDA / Triton / CuTe DSL / cuTile 中至少熟练两种、理解另外两种。
- 能写 PyTorch 可组合 custom op，并兼容 `torch.compile`。
- 能读 PTX/SASS/IR，用 Nsight Compute / Systems 给出优化证据。
- 能解释并实现简化的 KV Cache / PagedAttention / continuous batching runtime。
- 能理解并实测 NCCL collectives、TP/EP 与通信计算重叠；了解 NVSHMEM GPU-initiated communication。
- 能对 vLLM / CUTLASS / PyTorch / Triton / FlashInfer / SGLang 等至少一个大型项目进行源码级分析并完成有效贡献。

---

## 1. 每天 4 小时的固定模板

| 时间 | 内容 |
|---|---|
| 40 min | 官方文档 / 论文 / 源码精读，写 5–10 条结论 |
| 2 h | 编码：reference → kernel/runtime → tests |
| 40 min | benchmark / profiler / 参数扫描 |
| 40 min | 复盘：性能数字、失败原因、下一步、commit |

### Day 7 固定规则

Day 7 不追新知识，优先做四件事：**清测试、跑统一 benchmark、写 profiling 结论、整理 README/博客/PR**。如果本周主任务没完成，Day 7 先补主任务，不为了赶日历跳过关键能力。

---

## 2. Benchmark 与验收规范

以后所有「快了多少」都按统一口径记录，避免出现「GEMM 达到 cuBLAS 80%」但无法复现的问题。

- 固定记录：GPU、driver、CUDA、PyTorch、compiler flags、dtype、layout、shape、batch、warmup、迭代次数。
- correctness：至少与 PyTorch/cuBLAS/reference 对拍；低精度明确 `atol/rtol`；训练算子补 `gradcheck` 或对应梯度验证。
- latency：CUDA Event 或框架官方 benchmark 工具；必须 warmup；同步位置正确；报告 median/p50，必要时加 p95。
- throughput：GEMM 用 TFLOP/s；memory-bound kernel 用 effective bandwidth；serving 用 req/s、tok/s、TTFT、TPOT/ITL。
- profiler：至少给出一个「优化前 → 指标 → 推断 → 修改 → 优化后」的闭环。
- baseline：不要只比自己的 naive；应根据任务选择 cuBLAS/CUTLASS、PyTorch eager/SDPA、Triton、vLLM 等工业 baseline。
- 性能目标按 shape 划分，而不是设一个全局百分比。例如 GEMM 至少测小矩阵、方阵、LLM 常见 skinny/fat shape。

---

## 3. 12 个月总览

| 月份 | 主线 | 核心交付 |
|---|---|---|
| M1 | C++ / CUDA / GPU 基础 | P1 基础 kernel + benchmark 框架 |
| M2 | 并行原语 / GEMM / Roofline | P2 CUDA GEMM 优化报告 |
| M3 | Tensor Core / CUTLASS / CuTe | P3 多层次 GEMM + CuTe/CUTLASS 笔记 |
| M4 | DL 核心算子 / Triton / FlashAttention | P4 LLM kernel pack |
| M5 | PyTorch 集成 / torch.compile / Compiler | P5 compile-friendly custom-op library + mini compiler |
| M6 | KV Cache / PagedAttention / Scheduler | P6 Mini LLM Inference Runtime v1；开始试投 |
| M7 | FP8/INT8/FP4 / cuTile / Blackwell | 低精度 kernel pack + 现代架构报告 |
| M8 | MoE / Sampling / Speculative / 开源 | MoE mini-engine + 第一个有技术含量的 PR |
| M9 | NCCL / TP/EP / NVSHMEM | Multi-GPU Transformer Block |
| M10 | vLLM / Production Inference | P7 Production Inference Performance Report |
| M11 | 开源深挖 / 专项强化 | 1 个深度 PR + Capstone v2 |
| M12 | Capstone / 面试 / 正式投递 | 最终作品集 + 面试就绪 |

> 每个月份的逐周、逐日任务与验收标准，见对应 `Month_NN_*` 文件夹的 `README.md`。

---

## 4. 最终验收清单

### Kernel

- [ ] 不看模板写出 elementwise / transpose / reduction / softmax 基础 CUDA kernel。
- [ ] 能写并解释 tiled GEMM、register tiling、vectorization、async pipeline。
- [ ] 能解释 Tensor Core、TMA、Thread Block Cluster，以及 Hopper/Blackwell 差异。
- [ ] 至少熟练 CUDA + Triton/CuTe/cuTile 中另一套 DSL。
- [ ] 有 LLM kernel：RMSNorm、RoPE、Attention/FlashAttention、低精度或 MoE 中至少 4 类。

### Compiler / Framework

- [ ] 会用 `torch.library` 注册 custom op，做 opcheck/autograd。
- [ ] custom op 能与 `torch.compile` 正确组合。
- [ ] 能解释 Dynamo → AOTAutograd → Inductor 的基本链路。
- [ ] 能读 FX/Triton IR/PTX/SASS，并从编译产物定位一个性能问题。
- [ ] 做过至少一个 toy IR pass / fusion / codegen 项目。

### Runtime

- [ ] 能解释 prefill/decode 的性能差异。
- [ ] 写过 KV Cache manager 和简化 PagedAttention。
- [ ] 写过 continuous batching scheduler。
- [ ] 理解 chunked prefill、prefix caching、CUDA Graph、speculative decoding。
- [ ] 会用 TTFT/TPOT/throughput/concurrency 分析 serving。

### Distributed

- [ ] 能讲清 NCCL ring/tree、AllReduce/AllGather/ReduceScatter/AllToAll。
- [ ] 写过 TP Transformer block 或等价实验。
- [ ] 理解 MoE Expert Parallel 与 AllToAll。
- [ ] 能读 nsys timeline 判断通信/计算是否 overlap。
- [ ] 跑过 NVSHMEM demo，理解 GPU-initiated communication 的价值。

### 工程与求职

- [ ] 至少 5 个可复现 benchmark 项目，其中 2 个达到 portfolio 级质量。
- [ ] 1 个明星 Capstone：同时包含 kernel + runtime，最好再含 distributed。
- [ ] 至少 1 个真实开源技术 PR；以质量和 review 深度为 KPI，不把 merge 当唯一 KPI。
- [ ] 6–10 篇高质量技术笔记/博客，不追数量灌水。
- [ ] 每个简历性能数字都能回答：硬件？shape？dtype？baseline？计时方法？为什么快？

---

## 5. 推荐项目结构

```text
ai-kernel-systems-roadmap/
├── kernels/
│   ├── cuda/
│   ├── triton/
│   ├── cute_dsl/
│   └── cutile/
├── operators/
│   ├── gemm/
│   ├── normalization/
│   ├── attention/
│   ├── quantization/
│   └── moe/
├── compiler_lab/
│   ├── toy_ir/
│   └── torch_compile/
├── runtime/
│   ├── kv_cache/
│   ├── paged_attention/
│   ├── scheduler/
│   └── mini_server/
├── distributed/
│   ├── nccl/
│   ├── tensor_parallel/
│   ├── expert_parallel/
│   └── nvshmem/
├── benchmarks/
├── profiling/
├── tests/
├── docs/
└── README.md
```

> 本仓库已按此创建对应目录骨架，各目录内含占位 `README.md` 说明用途与对应月份。

---

## 6. 2026 官方资料优先级

以下资料应优先于二手博客。二手文章用于补直觉，不作为版本/接口/架构事实的唯一依据。

### NVIDIA CUDA / Architecture

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Compute Capabilities: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/
- Nsight Systems: https://docs.nvidia.com/nsight-systems/

### CUTLASS / CuTe DSL

- CUTLASS: https://docs.nvidia.com/cutlass/latest/
- CUTLASS Overview / CuTe DSL: https://docs.nvidia.com/cutlass/latest/overview.html
- CuTe DSL docs: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html

### cuTile Python

- cuTile Python: https://docs.nvidia.com/cuda/cutile-python/
- Quickstart: https://docs.nvidia.com/cuda/cutile-python/quickstart.html
- Compilation/AOT: https://docs.nvidia.com/cuda/cutile-python/compilation.html

> 2026-09 的 cuTile 文档已覆盖 8.x、9.x、10.x、11.x、12.x GPU，且 1.6.0 发布于 2026-09-09。具体安装要求和支持矩阵应以当时官方文档为准。

### PyTorch

- Custom Operators: https://docs.pytorch.org/docs/main/library.html
- User-defined Triton kernels with `torch.compile`: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- `torch.compile`: https://docs.pytorch.org/docs/stable/torch.compiler.html

> 现代 PyTorch custom op 优先学习 `torch.library` / `triton_op` / `register_autograd` / `opcheck`。`torch.autograd.Function` 仍值得理解，但不要把它当唯一集成方式。

### vLLM / Serving

- vLLM docs: https://docs.vllm.ai/en/stable/
- Optimization and Tuning: https://docs.vllm.ai/en/latest/configuration/optimization/
- Data Parallel Deployment: https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
- Expert Parallel Deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

> 当前 vLLM 已覆盖 PagedAttention、continuous batching、chunked prefill、prefix caching、CUDA Graph、torch.compile、多种低精度、speculative decoding，以及 TP/PP/DP/EP/CP。学习 runtime 时应直接以源码和官方文档为主。

### Distributed

- NCCL docs: https://docs.nvidia.com/deeplearning/nccl/
- NVSHMEM: https://docs.nvidia.com/nvshmem/

> NVSHMEM 的重要价值是允许从 CUDA kernel / CUDA stream 发起更细粒度 GPU-GPU 通信，适合理解 GPU-initiated communication 与通信计算融合。

---

## 7. 需要主动降级或延后的内容

学习计划不是越满越好。遇到时间不足时，按下面顺序删减：

1. **先删「为了完整而完整」的传统 Conv 深挖**，只保留 implicit GEMM 思想。
2. 再删部分老式 WMMA API 细节，但保留 Tensor Core 数据流和 PTX/SASS 认知。
3. 2:4 sparsity 降为选修，除非目标 JD 明确要求。
4. 不删：GEMM、Attention、KV/PagedAttention、torch.compile、CuTe/cuTile、NCCL、vLLM、profiling。
5. 不为了追赶日期跳过 correctness/benchmark；一个高质量项目 > 五个半成品。

---

## 8. 最重要的执行原则

**每个月至少交付一个「别人可以验证」的东西。**

不要写：

> 「学习了 CUDA、Triton、vLLM。」

要能写成：

> 「在 RTX 4090 上实现并优化 BF16 RMSNorm/Attention kernel；针对指定 batch/seq/head-dim shape 与 PyTorch/Triton baseline 对比，使用 Nsight Compute 定位 memory throughput 与 warp stall，优化后 p50 latency 降低 X%，并通过误差与梯度测试。」

真正决定你能不能进入 AI Kernel / AI Systems 团队的，不是知识点数量，而是你能不能完成：

**问题定义 → correctness → performance model → implementation → profiling → optimization → system integration → reproducible evidence。**
