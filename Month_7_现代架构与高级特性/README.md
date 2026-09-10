# Month 7：现代架构与高级特性（Phase 2 起航）

> **主题：** 进入 Hopper/Blackwell 时代，掌握 TMA、wgmma、Thread Block Cluster、CUDA Graph、异步拷贝等现代特性。
> **前置：** 完成 Phase 1（M1–M6），达成 checkpoint。
> **重要提醒：** 这些特性多在 Hopper（H100 / sm_90）上才能实测；**你的 4090（sm_89）不支持 TMA / wgmma / Thread Block Cluster**。策略是「概念 + 读源码 + 代码结构」为主，**按需租 1–2 小时 H100 做少量实测**，其余用 `-arch=sm_90` 编译尝试（能编译过就够）；`cp.async`（Ampere 起）在 4090 上可以直接实测。
> **月度目标：** 理解并能在代码里用 TMA/wgmma/cluster 的结构；掌握 CUDA Graph 与异步；写一篇现代特性博客。

## 月度里程碑

- ✅ 理解 Hopper 引入 TMA / wgmma / Thread Block Cluster 的动机与代码结构。
- ✅ 掌握 `cp.async`（Ampere 起）与 TMA 异步拷贝，理解异步流水线。
- ✅ 会用 CUDA Graph 减少 kernel 启动开销，理解多流与事件。
- ✅ 写一篇《Hopper TMA/wgmma 实测》博客。

---

## Week 1：Hopper 架构与 TMA

**周目标：** 理解 Hopper 架构变化与 TMA（Tensor Memory Accelerator）异步拷贝机制。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Hopper 架构概览 | H100 的 SM、warp scheduler、Tensor Core（warpgroup 级 wgmma）、TMA、thread block cluster | 读 NVIDIA H100 白皮书架构章节 | 能画出 H100 SM 结构并对比 A100 |
| Day 2 | 为什么需要 TMA | 传统拷贝靠线程 + 寄存器循环，浪费线程资源；TMA 用专用引擎异步搬 tensor | 读 TMA 官方博客/文档 | 能说清 TMA 解决什么问题 |
| Day 3 | `cp.async`（Ampere 基础） | `cp.async` 全局→共享内存异步拷贝、`cp.async.wait_group`、`cp.async.commit` | 用 `cp.async` 重写 GEMM 的 tile 加载（4090 直接实测，`-arch=sm_89`） | 理解异步拷贝流水线 |
| Day 4 | TMA 编程模型 | `cuda::memcpy_async`、`cuda::barrier`、`cuda::pipeline`、`cuda::tensor_map` | 读 CUTLASS 里 TMA 用法，跑通一个官方示例（需 A100/H100） | 能看懂 TMA 代码结构 |
| Day 5 | TMA 与共享内存 | TMA 直接全局→共享，减少寄存器占用与指令 | 对比「线程拷贝 vs TMA」的资源占用 | 能说清 TMA 的收益 |
| Day 6 | 编译与 arch | `-arch=sm_90` 编译、`__CUDA_ARCH__` 条件编译、`cudaDeviceProp` 查特性 | 写一个 `#if __CUDA_ARCH__ >= 900` 的条件编译骨架 | 能写「4090 可编译、H100 走新路径」的兼容代码 |
| Day 7 | 复习 + 综合 | 整理 TMA/异步拷贝笔记 | 完成 Week 1 笔记 | 交出 TMA 笔记 |

**Week 1 自检：** 能说清 TMA 动机与代码结构；会用 `cp.async`；能写 `__CUDA_ARCH__` 条件编译。

---

## Week 2：wgmma、Thread Block Cluster

**周目标：** 理解 warpgroup 级矩阵乘（wgmma）与 Thread Block Cluster（分布式共享内存）。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | warpgroup 与 wgmma | warpgroup = 4 个 warp 协作；`wgmma.mma_async` 一步完成大 tile 矩阵乘 | 读 wgmma PTX 文档 | 理解 wgmma 与 wmma 的区别 |
| Day 2 | wgmma 代码结构 | 在 CUTLASS Hopper GEMM 里看 wgmma 用法 | 读 CUTLASS `sm90` GEMM 源码 | 能看懂 wgmma 调用 |
| Day 3 | Thread Block Cluster | 多个 block 组成 cluster，共享分布式共享内存（DSMEM） | 读 cluster 官方文档 + 示例 | 理解 cluster 的价值（更大 tile、跨 block 通信） |
| Day 4 | DSMEM 编程 | `cluster.sync()`、`map_shared_rank`、跨 block 读共享内存 | 跑 cluster 官方示例（需 H100） | 能写一个最小 cluster 程序 |
| Day 5 | Hopper GEMM 全貌 | TMA + wgmma + cluster 组合成 Hopper 级 GEMM | 用 CUTLASS 跑 H100 GEMM（租用），记录 GFLOPs | 看到 H100 级性能 |
| Day 6 | Blackwell 前瞻 | B200 的第五代 Tensor Core、FP4/FP6、更多特性 | 读 Blackwell 白皮书概览 | 了解下一代趋势 |
| Day 7 | 复习 + 综合 | 整理 wgmma/cluster 笔记 | 完成 Week 2 笔记 | 交出 wgmma/cluster 笔记 |

**Week 2 自检：** 能说清 wgmma 与 wmma 差异；理解 Thread Block Cluster 与 DSMEM；能在 CUTLASS 里定位 Hopper 特性。

---

## Week 3：CUDA Graph 与多流

**周目标：** 掌握 CUDA Graph 减少启动开销、多流并发、事件同步，理解计算与拷贝重叠。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | kernel 启动开销 | 每个 kernel 有 CPU→GPU 启动延迟；小 kernel 多时开销显著 | 测 1000 个小 kernel 的累计启动开销 | 能说清为什么小 kernel 慢 |
| Day 2 | CUDA Graph | `cudaGraphCreate`/`cudaGraphLaunch`、捕获（capture）、节点依赖 | 把一串 kernel 捕获成 Graph，对比普通启动 | 理解 Graph 如何摊薄开销 |
| Day 3 | Graph 进阶 | `cudaStreamBeginCapture`、动态 Graph、Graph 更新 | 写一个用 Stream Capture 的例子 | 会用捕获式构建 Graph |
| Day 4 | 多流并发 | 多个 stream 并行执行、`cudaMemcpyAsync`、事件 `cudaEventRecord` | 写两个 stream 分别跑 kernel，看并发 | 理解多流并行 |
| Day 5 | 拷贝与计算重叠 | 用流让 H2D 拷贝与上一 kernel 计算重叠 | 写「拷贝/计算重叠」流水线 | 能说清 overlap 的收益与条件 |
| Day 6 | 异步进阶 | pinned memory、`cudaMallocAsync`、`cudaMemPool` | 用 pinned memory + 多流优化一个数据搬运场景 | 理解异步内存分配 |
| Day 7 | 复习 + 综合 | 整理 CUDA Graph/多流笔记 | 完成 Week 3 笔记 | 交出 Graph/多流笔记 |

**Week 3 自检：** 会用 CUDA Graph 减少启动开销；能用多流并发；能实现拷贝与计算重叠。

---

## Week 4：综合实验与博客

**周目标：** 综合应用现代特性做一次实测，沉淀成博客。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 租用 H100 实测 | 租 1–2 小时 H100 云实例，跑自己 M3 的 GEMM | 在 H100 上跑 GEMM，记录性能 | 有自己的 H100 数字 |
| Day 2 | 现代 GEMM 实测 | 用 CUTLASS 在 H100 跑 GEMM，对比自己 M3 版 | CUTLASS H100 vs 自写 GEMM 对比 | 拿到 Hopper 级对比 |
| Day 3 | 写博客 | 《Hopper TMA/wgmma 实测：从概念到代码》 | 整理成博客 | 博客草稿完成 |
| Day 4 | 完善博客 | 补性能数字、代码片段、架构图 | 发布博客 | 博客发布 |
| Day 5 | 持续开源 | 回看 M6 的 PR、找新的算子 issue | 提交/推进 1 个 PR | 开源持续 |
| Day 6 | P6 规划 | 规划明星项目 P6（Flash Attention 复现/优化）范围 | 写 P6 需求与里程碑 | P6 计划清晰 |
| Day 7 | 月度复习 | 复盘 M7；完成月度自检 | 完成自检 + 笔记归档 | 交出 M7 成果 |

**M7 月度自检清单：**
- [ ] 能说清 TMA / wgmma / Thread Block Cluster 的动机与代码结构。
- [ ] 会用 `cp.async`，能写 `__CUDA_ARCH__` 条件编译。
- [ ] 会用 CUDA Graph、多流、实现拷贝与计算重叠。
- [ ] 在 H100 上做过一次实测，有性能数字。
- [ ] 博客《Hopper 特性实测》已发布。

## 本月背书行动（在职）

- **发《Hopper TMA/wgmma 实测》博客**：展示你懂现代架构，是区别于普通 CUDA 入门者的关键证明。
- **持续开源**：推进既有 PR + 找新 issue。
- 产出口径：**1 篇现代特性博客 + 1 个 PR + H100 实测数字**。

## Month 7 参考资源

- NVIDIA H100 / Blackwell 白皮书（架构章节）
- CUDA C++ Programming Guide：`cp.async`、CUDA Graph、Thread Block Cluster 章节
- CUTLASS `examples/` 的 sm90 示例
- NVIDIA 博客：TMA、Thread Block Cluster、CUDA Graphs 系列
