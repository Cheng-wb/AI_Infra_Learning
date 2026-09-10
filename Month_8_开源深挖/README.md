# Month 8：开源深挖

> **主题：** 深入 CUTLASS 源码，向 vLLM / xFormers 贡献，复现/优化 Flash Attention，做出明星项目 P6。
> **前置：** 完成 M7（现代特性有概念与实测）。
> **月度目标：** 能吃透一个大型 CUDA 代码库（CUTLASS）；做出 **1 个有分量的性能优化 PR**；启动/推进明星项目 P6。

## 月度里程碑

- ✅ 精读 CUTLASS 核心源码（tile 抽象、mainloop、epilogue、Hopper 路径）。
- ✅ 在 vLLM / xFormers / PyTorch 里做出 1 个有分量的性能优化 PR。
- ✅ 复现或深度优化 Flash Attention（P6 明星项目）。
- ✅ 发布 1 篇深度源码/优化博客。

---

## Week 1：CUTLASS 源码精读

**周目标：** 从「会用 CUTLASS」升级到「能读 CUTLASS」，理解其模板元编程与分层抽象。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 总体架构 | CUTLASS 分层：`gemm/` 主循环、`collective`、`epilogue`、`sm80`/`sm90` 后端 | 画出 CUTLASS GEMM 的模块依赖图 | 能说出各层职责 |
| Day 2 | Mainloop 精读 | tile 循环、`cp.async`/TMA 加载、`mma` 调用、双缓冲 | 逐段读一个 sm80 GEMM mainloop | 能讲清一次 tile 迭代 |
| Day 3 | Epilogue 精读 | 累加器→输出、bias/激活/epilogue 融合 | 读 epilogue 的融合写法 | 理解 epilogue 如何接算子 |
| Day 4 | 模板元编程 | `traits`、`typename...`、编译期分派、`static_if` | 手写一个小的 CUTLASS 风格 traits | 能写简单模板元编程 |
| Day 5 | Hopper 路径 | sm90 的 TMA + wgmma + cluster 组合源码 | 读 sm90 GEMM 关键路径 | 能定位 Hopper 特性在源码里的位置 |
| Day 6 | 自定义算子接入 | 用 CUTLASS 的 epilogue 自定义一个融合算子 | 基于 CUTLASS 写一个自定义 epilogue | 能用 CUTLASS 做定制 |
| Day 7 | 复习 + 综合 | 写 CUTLASS 源码笔记 | 完成 Week 1 笔记 | 交出 CUTLASS 源码笔记 |

**Week 1 自检：** 能画出 CUTLASS 分层架构；能讲清 mainloop/epilogue；能写简单 traits。

---

## Week 2：vLLM / xFormers 贡献

**周目标：** 在真实推理/训练框架里找到算子优化点并贡献。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | vLLM 算子概览 | vLLM 的 PagedAttention、fused 算子、CUDA kernel 目录 | 读 vLLM `csrc/` 的 kernel 目录结构 | 了解 vLLM 算子组织 |
| Day 2 | 找优化点 | 翻 vLLM 的 issue（performance 标签）、benchmark 热点算子 | 跑通 vLLM，profiling 一个热点 kernel | 锁定一个可优化点 |
| Day 3 | xFormers 概览 | xFormers 的 memory-efficient attention、算子结构 | 读 xFormers attention 实现 | 了解其算子 |
| Day 4 | 写性能优化 patch | 针对锁定的热点算子做优化（tile/向量化/融合/occupancy） | 写 patch + 本地 benchmark | patch 有可测的性能提升 |
| Day 5 | 提交 PR | 按 `CONTRIBUTING.md` 提交，附 benchmark 对比 | 提交性能优化 PR | PR 提交并进 review |
| Day 6 | 响应 review | 根据 reviewer 意见修改、跑 CI | 迭代到 CI 通过 | PR 逼近 merge |
| Day 7 | 复习 + 综合 | 记录「找点 → 优化 → 提交」流程 | 完成 Week 2 笔记 | 交出 1 个性能 PR |

**Week 2 自检：** 能在 vLLM/xFormers 找到并优化一个热点算子；提交带 benchmark 的 PR；能响应 review。

---

## Week 3：Flash Attention 复现与优化（P6）

**周目标：** 把 M5 的简化 Flash Attention 升级为有分量的明星项目。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 明确 P6 范围 | 目标：一个「分块 + online softmax + causal + backward」的完整 FA，或对开源 FA 做性能优化 | 写 P6 需求文档 | P6 目标清晰 |
| Day 2 | 完整前向 | 实现完整 FA 前向（含 causal mask、dropout） | 写完整前向，对拍 PyTorch | 前向对拍通过 |
| Day 3 | 反向实现 | FA backward 的梯度重算（rematerialization）与分块 | 写反向，gradcheck | 反向正确 |
| Day 4 | 性能优化 | 用 Tensor Core（`tl.dot`）、调 block 尺寸、双缓冲 | 优化到接近或超过 SDPA | 有性能数字 |
| Day 5 | 与现代特性结合 | 尝试接入 `cp.async`/TMA（可选，A100/H100） | 做一次现代特性版 | 可选加分项 |
| Day 6 | benchmark + 文档 | 全规模 benchmark（naive/SDPA/自写 FA），写 README | 完成 benchmark 对比 + README | P6 可对外展示 |
| Day 7 | 复习 + 综合 | 写《Flash Attention 从实现到优化》博客 | 完成 P6 + 博客 | 交出 P6 明星项目 + 博客 |

**Week 3 自检：** 完整 FA（前向+反向）对拍通过；性能对标 SDPA；有 benchmark 与 README。

---

## Week 4：明星项目打磨与博客

**周目标：** 把 P6 打磨成简历核心项目，并沉淀博客。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 代码质量 | 重构 P6：接口、命名、错误检查、注释 | 重构代码 | 代码达到开源标准 |
| Day 2 | 单测与 CI | 加单测、写 CI（GitHub Actions 跑测试） | 配置 CI | CI 绿灯 |
| Day 3 | 性能曲线 | 画「seq len vs 速度」「显存 vs 方法」曲线 | 完善 benchmark 可视化 | 图表齐备 |
| Day 4 | 博客发布 | 《Flash Attention 从实现到优化》发布 | 发布博客 | 博客上线 |
| Day 5 | 推广 | 发到社区/知乎/Reddit/HN，收反馈 | 推广 P6 | 获得外部反馈 |
| Day 6 | 开源 PR 跟进 | 跟进 Week 2 的 PR 到 merge | 推进 PR merge | PR merged 或接近 |
| Day 7 | 月度复习 | 复盘 M8；月度自检 | 完成自检 + 归档 | 交出 M8 成果 |

**M8 月度自检清单：**
- [ ] 能讲清 CUTLASS 分层架构与 mainloop/epilogue。
- [ ] 做出 1 个有分量的性能优化 PR（vLLM/xFormers/PyTorch）。
- [ ] P6 明星项目（完整 FA）完成，对拍 + benchmark + README 齐备。
- [ ] 发布 1 篇深度博客。

## 本月背书行动（在职）

- **1 个有分量的性能优化 PR**（这是在职者最硬的算子岗背书，目标 merge）。
- **P6 明星项目**：完整 Flash Attention 实现，成为简历核心项目。
- **1 篇深度博客** + CUTLASS 源码笔记。
- 产出口径：**1 个性能 PR + P6 项目 + 1 篇博客**。

## Month 8 参考资源

- CUTLASS 源码（`include/cute/`、`include/cutlass/gemm/`）
- vLLM / xFormers 源码与 `CONTRIBUTING.md`
- FlashAttention 2/3 论文 + 开源实现（Dao-AILab/flash-attention）
