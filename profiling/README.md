# profiling/

Nsight Compute（`ncu`）与 Nsight Systems（`nsys`）的 profiling 报告与证据。

- 至少给出一条「优化前 → 指标 → 推断 → 修改 → 优化后」的闭环。
- 记录 memory throughput、warp stall、occupancy、CPU/GPU bubble、通信/计算 overlap 等关键指标。
- 报告要能被陌生人复现（含命令、环境、截图/导出文件）。
