# benchmarks/

统一 benchmark 脚本与结果。

- 口径固定记录：GPU、driver、CUDA、PyTorch、compiler flags、dtype、layout、shape、batch、warmup、迭代次数。
- latency 用 CUDA Event，warmup + median/p50（必要时 p95）；throughput 用 TFLOP/s、effective bandwidth 或 serving 指标。
- baseline 选 cuBLAS/CUTLASS、PyTorch eager/SDPA、Triton、vLLM 等工业基线，性能目标按 shape 划分。

产出：可复现的 CSV / 曲线 / 热力图。
