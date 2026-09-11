# Month 4：LLM 核心算子、Triton 与 FlashAttention

> **月目标：** 从 GEMM 转向现代 LLM 最常见 kernel，并学习第二种高生产力 DSL。
> **Must：** RMSNorm、RoPE、Softmax、GQA/MQA Attention、Triton。
> **Should：** FlashAttention forward/backward。
> **Stretch：** variable-length / paged KV-friendly attention。

## Week 1 — RMSNorm / RoPE / Softmax

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | RMSNorm | CUDA forward + PyTorch reference | 误差/带宽报告 |
| 2 | LayerNorm vs RMSNorm | 写两者并比较 reduction 成本 | 能解释差异 |
| 3 | RoPE | CUDA rotary embedding | 支持常见 interleaved/layout |
| 4 | Stable Softmax | max+sum 两阶段/warp 版 | 大值输入不 NaN |
| 5 | Online Softmax | 单遍/块级状态合并 | 推导 rescale 公式 |
| 6 | Fusion | residual + RMSNorm 或 bias+activation | 与拆分版本对比 |
| 7 | 周收口 | LLM normalization pack | tests + benchmark |

## Week 2 — Attention / GQA / MQA / KV Layout

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | MHA | PyTorch/CPU reference + naive CUDA path | shape 定义清楚 |
| 2 | Causal / mask | causal attention | 数值正确 |
| 3 | MQA/GQA | head mapping 实现 | 与 PyTorch 对拍 |
| 4 | KV Cache layout | contiguous KV cache 原型 | append/read 正确 |
| 5 | Prefill vs Decode | 分别 benchmark attention | 能说明 compute/memory 特征 |
| 6 | Memory traffic | 测中间张量与带宽 | 找到 decode bottleneck |
| 7 | 周收口 | attention 数据流文档 | 画 Q/K/V/KV-cache 生命周期 |

## Week 3 — Triton

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Triton model | vector add/softmax | 理解 program_id/block model |
| 2 | Triton reduction | RMSNorm/Softmax 二选一 | 对拍 CUDA |
| 3 | Triton GEMM | `tl.dot` GEMM | 解释 Tensor Core 映射 |
| 4 | Autotune | configs/key/num_warps | 形成 shape-specific config |
| 5 | Triton IR | dump TTIR/TTGIR/LLVM/PTX | 能找到关键 lowering |
| 6 | Triton vs CUDA | 同算子 benchmark | 比开发成本与性能 |
| 7 | 周收口 | Triton kernel pack | 统一 bench |

## Week 4 — FlashAttention

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | FA IO-aware 思想 | 推导 tiled online softmax | 能白板讲清 |
| 2 | Forward | Triton/CUDA 实现简化 FA forward | 对拍 SDPA |
| 3 | Causal/GQA | 加 causal + GQA | correctness |
| 4 | Backward | 实现或复现 backward | 梯度验证 |
| 5 | Benchmark | seq_len/head_dim/dtype sweep | latency+memory 曲线 |
| 6 | Profiling | 对比 SDPA/naive/自写 | 解释优势与差距 |
| 7 | 月收口 | 发布 P4 LLM Kernel Pack | 至少 4 类 kernel + 报告 |

## Month 4 自检

- [ ] 手写 RMSNorm / RoPE / Stable+Online Softmax，大值输入不 NaN，误差/带宽有报告。
- [ ] 手写 MHA + causal + MQA/GQA attention，与 PyTorch 对拍，能说明 prefill/decode 特征。
- [ ] 用 Triton 写 reduction + `tl.dot` GEMM，能 dump TTIR/TTGIR/LLVM/PTX 并解释 Tensor Core 映射。
- [ ] 实现简化 FlashAttention forward（+ backward），对拍 SDPA，能白板讲 IO-aware 思想。
- [ ] P4 LLM Kernel Pack 已发布（至少 4 类 kernel + benchmark + 报告）。

> 完成后进入 [Month 5 — PyTorch 集成 / torch.compile / Compiler](../Month_05_PyTorch集成与编译器/README.md)。
