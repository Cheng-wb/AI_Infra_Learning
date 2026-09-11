# Month 1：C++、CUDA 与 GPU Performance 基础

> **月目标：** 从「能写 C++」升级到「能可靠写、测、debug CUDA kernel」。
> **Must：** elementwise / transpose / reduction 基础版，统一 correctness/benchmark 框架。
> **Should：** 学会 Nsight Compute 基础指标。
> **Stretch：** 给 P1 加 CI 和 sanitizer/debug workflow。

## Week 1 — C++ for Kernel Engineering

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | 值语义、类型、编译 | 用 C++ 重写 Python 小程序；开启 `-Wall -Wextra -O2` | 能解释栈/堆、整数溢出、编译警告 |
| 2 | 指针、引用、const | 写数组遍历、swap、span 风格接口 | 能解释 pointer arithmetic 与 const correctness |
| 3 | RAII 与资源管理 | 写 CPU Buffer/文件句柄 RAII 类 | 无泄漏；理解 copy/move |
| 4 | STL 与数据布局 | vector/array/span；AoS vs SoA 小实验 | 能解释布局对 cache/coalescing 的影响 |
| 5 | 模板与 constexpr | 写 dtype 泛型向量运算和 compile-time shape helper | int/float/double 测试通过 |
| 6 | CMake / 编译链接 | `.h/.cpp` 拆分；静态库 + test target | 一条命令构建测试 |
| 7 | 周收口 | 写 Matrix/Buffer 小模块 + tests + README | Git tag `m1-w1`，测试全绿 |

> Week 1 详细学习笔记见 `Week_1/` 目录（Day1–Day7）。

## Week 2 — CUDA Programming Model

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Host/Device 与启动模型 | vector add；线程索引可视化 | 理解 grid/block/thread 与 launch |
| 2 | CUDA memory/API | `cudaMalloc/cudaMemcpy/cudaFree` + error macro | 每次 API/kernel launch 有错误检查 |
| 3 | Warp/SIMT | branch divergence 对比实验 | 能解释 warp divergence 发生条件 |
| 4 | SM / latency hiding | 扫 block size 64/128/256/512 | 能解释 occupancy 不等于性能 |
| 5 | CUDA Event | 实现 warmup + median benchmark helper | 计时不包含错误同步/初始化 |
| 6 | Streams/events 入门 | 两个 stream + event dependency | 正确输出依赖时序 |
| 7 | 周收口 | SAXPY + add/mul/relu 统一 API | correctness + latency 表 |

## Week 3 — Memory Hierarchy 与访存优化

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Coalescing | 连续 vs stride global load | 记录 effective bandwidth |
| 2 | Shared memory | block cache + reuse 小实验 | 能解释 shared memory 的价值 |
| 3 | Bank conflict | transpose 32×32 与 padding 32×33 | profiler/时间展示冲突差异 |
| 4 | Registers / spill | 人为提高局部变量与 unroll | 用 ptxas 看 register/spill |
| 5 | Alignment / vectorized IO | float2/float4 load/store | 正确处理尾部和对齐 |
| 6 | L2 / readonly / restrict | `const __restrict__` 与访问模式实验 | 用证据而非猜测说明变化 |
| 7 | 周收口 | transpose 三版本整理 | README 含布局图和 benchmark |

## Week 4 — Reduction、Profiling 与 P1

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Reduction tree | shared-memory reduction | CPU 对拍；任意 N |
| 2 | Warp shuffle | `__shfl_down_sync` reduction | 与 shared 版比较 |
| 3 | Multi-block | partial reduction + second pass | 大数组正确 |
| 4 | Nsight Compute | 跑 elementwise/transpose/reduction | 找到 memory throughput/stall/occupancy |
| 5 | Debugging | compute-sanitizer + 越界/竞态案例 | 能定位两类错误 |
| 6 | P1 整合 | `src/ tests/ bench/ docs/` 工程结构 | 一条命令测试和 benchmark |
| 7 | 月收口 | 发布 P1 + 第一篇性能笔记 | 有复现命令、环境、数字 |

## Month 1 自检

- [ ] 不看模板写出 elementwise / transpose / reduction 基础 CUDA kernel（含错误检查 + CPU 对拍）。
- [ ] 能解释 Thread/Block/Grid、warp/SM/SIMT、occupancy 与 latency hiding。
- [ ] 能画出内存层次图，解释 coalescing、bank conflict、register spill、vectorized IO。
- [ ] 有统一的 correctness/benchmark 框架，计时口径正确（warmup + median）。
- [ ] 用 Nsight Compute 跑过基础 kernel，能读 memory throughput/stall/occupancy。
- [ ] P1 基础 kernel + benchmark 框架已发布（含复现命令、环境、数字）。

> 完成后进入 [Month 2 — 并行原语 / GEMM / Roofline](../Month_02_并行原语与GEMM/README.md)。
