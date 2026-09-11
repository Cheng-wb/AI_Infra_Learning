# kernels/

最底层的 kernel 实现，按编程模型 / DSL 分类。所有 kernel 都遵守根 README 第 2 节的 benchmark 与验收规范（correctness + benchmark + profiler evidence + reproducibility）。

| 子目录 | 内容 | 主要月份 |
|---|---|---|
| `cuda/` | 手写 CUDA C++ kernel（elementwise/reduction/transpose/GEMM/attention…） | M1–M4、M7、M8 |
| `triton/` | Triton kernel（softmax/RMSNorm/GEMM/FlashAttention…） | M4、M5 |
| `cute_dsl/` | CuTe / CuTe DSL（layout algebra、tiled GEMM、CUTLASS 示例） | M3、M7 |
| `cutile/` | cuTile Python（tile model、matmul、autotune、AOT） | M7 |

每个子目录内以「算子名 / 版本」为单位组织，配套 reference 实现、对拍测试与 benchmark。
