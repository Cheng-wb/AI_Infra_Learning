# Month 7：Low Precision、cuTile 与 Blackwell

> **月目标：** 补齐 2026 年高价值低精度和 NVIDIA 新一代 DSL/架构。
> **Must：** BF16/FP16/FP8/INT8 误差与 kernel；cuTile 基础。
> **Should：** block-scaled FP8/FP4、Blackwell 编程模型。
> **Stretch：** B200/GB200 上实测 FP4/Blackwell kernel。

## Week 1 — Quantization Fundamentals

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Numeric formats | FP32/BF16/FP16/FP8/INT8/FP4 对照实验 | 范围/ULP/误差表 |
| 2 | Symmetric quant | per-tensor INT8 quant/dequant | reference 正确 |
| 3 | Per-channel/group | weight group quantization | 误差明显优于粗粒度场景 |
| 4 | Weight-only | quantized linear + on-the-fly dequant | correctness |
| 5 | GPTQ/AWQ 思想 | 读算法并复现实验级量化 | 能解释离线量化与 kernel 分工 |
| 6 | Calibration | activation scale/outlier 实验 | 有误差分布图 |
| 7 | 周收口 | quantization notebook/report | 不只记录「位宽更低更快」 |

## Week 2 — FP8 / INT8 GEMM 与 Fusion

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | INT8 GEMM | int8 input + int32 accumulation | 对拍 baseline |
| 2 | FP8 scale | E4M3/E5M2 scaling 实验 | 解释 scale 更新 |
| 3 | FP8 matmul | 用 PyTorch/CUTLASS 支持路径 benchmark | 性能+误差 |
| 4 | Dequant fusion | dequant + GEMM/epilogue 融合 | 减少额外 kernel |
| 5 | Block scaling | 研究 MX/NVFP4 类 block scale | 写 block-scale reference |
| 6 | Shape sweep | prefill/decode/MoE shape | 找低精度收益条件 |
| 7 | 周收口 | Low Precision Kernel Pack v1 | benchmark matrix |

## Week 3 — cuTile Python

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | cuTile execution model | vector add / elementwise | 理解 tile vs thread model |
| 2 | Tile data model | reshape/broadcast/reduction | 完成小算子 |
| 3 | cuTile matmul | 写/改 matmul 示例 | correctness |
| 4 | Specialization | static shape/divisibility hints | 查看 JIT 行为 |
| 5 | Autotuning | tile config sweep | 找优选配置 |
| 6 | AOT/export | 导出 cubin/TileIR（环境支持时） | 能解释 JIT/AOT |
| 7 | 周收口 | CUDA/Triton/CuTe/cuTile 对比 | 形成 DSL 选择指南 |

## Week 4 — Blackwell Architecture

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Compute capability 10.x | 阅读 CUDA compute capability/tuning | 整理 Hopper→Blackwell 差异 |
| 2 | Tensor Core/FP4 | 阅读 Blackwell low-precision 路径 | 能解释 FP4 使用场景 |
| 3 | Cluster / DSMEM | 复习 cluster 并看 Blackwell 变化 | 画 GPC/cluster 数据流 |
| 4 | Cluster Launch Control | 阅读 work stealing 机制 | 能解释解决的负载均衡问题 |
| 5 | CUTLASS Blackwell | 跑/读 sm100/CuTe DSL example | 定位核心代码 |
| 6 | B200 实测 | 若有资源，跑 FP8/FP4/GEMM；否则做 compile/source study | 有实测或替代证据 |
| 7 | 月收口 | 现代架构性能报告 | 明确 4090/H100/B200 能力边界 |

## Month 7 自检

- [ ] 有 FP32/BF16/FP16/FP8/INT8/FP4 的范围/ULP/误差对照表，能解释 per-tensor/per-channel/group quant 差异。
- [ ] 写过 INT8 GEMM（int32 accumulation）与 FP8 scale matmul，能融合 dequant 减少额外 kernel。
- [ ] 能用 cuTile 写 elementwise/matmul，理解 tile vs thread model、JIT/AOT，形成 DSL 选择指南。
- [ ] 能解释 Hopper→Blackwell（compute capability 10.x）差异、FP4 使用场景、Cluster Launch Control。
- [ ] 低精度 kernel pack + 现代架构性能报告已发布（明确 4090/H100/B200 能力边界）。

> 完成后进入 [Month 8 — MoE / Sampling / Speculative / 开源](../Month_08_MoE与解码/README.md)。
