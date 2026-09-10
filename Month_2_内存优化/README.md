# Month 2：内存优化与核心数据算子

> **主题：** 系统掌握共享内存与访存优化，实现并行编程三大原语——**归约（Reduction）、前缀和（Scan）、转置（Transpose）**，理解 Bank Conflict 与 Occupancy。
> **前置：** 完成 M1（能写并验证基础 kernel、有 CPU 对拍框架）。
> **月度目标：** 三件套算子全部通过 CPU 对拍并达到较好带宽；理解「为什么快/为什么慢」；完成 **P1 基础算子库**。

## 月度里程碑

- ✅ 用共享内存消除 Bank Conflict，完成矩阵转置并对比三版本性能。
- ✅ 实现高性能归约（naive → shared-memory → warp shuffle → 多 block）。
- ✅ 实现前缀和 Scan（Hillis-Steele / Blelloch）并通过 CPU 对拍。
- ✅ 掌握 float4 向量化、`#pragma unroll`、Occupancy 与 launch config。
- ✅ 完成 P1 基础算子库（element-wise + reduction + transpose + scan + 统一对拍与 benchmark）。

---

## Week 1：共享内存与 Bank Conflict

**周目标：** 吃透共享内存的读写速度来源与 Bank Conflict 机制，学会用 padding 消除冲突。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 共享内存深入 | `__shared__` 声明、静态/动态分配、每 SM 的容量（4090/sm_89 约 128KB，可配更高）、生命周期 = block | 写一个用共享内存缓存 block 内数据的 kernel，与全局内存版计时对比 | 能解释共享内存「低延迟、bank 化」的两大特性 |
| Day 2 | Bank Conflict 原理 | 32 个 bank、每 bank 4 字节宽；同 bank 同地址广播、同 bank 不同地址串行（n-way conflict） | 写一个 stride=1 与 stride=32 的共享内存读，观察冲突 | 能手算一段访问模式的 conflict 数 |
| Day 3 | 转置 naive 版 | 全局内存读列不合并 → 用共享内存中转，行读列写 | 写 `transpose` 共享内存版：`tile[tid]` 读入，转置后写出 | 跑通 + 对拍，记录 naive 版带宽 |
| Day 4 | 消除 Bank Conflict | `__shared__ T tile[32][33]` 加 1 列 padding，让同列错开 bank | 对比加 padding 前后转置速度 | 能解释 padding 为何消除 32 路冲突 |
| Day 5 | 共享内存做 tile | 矩阵乘法雏形：每个 block 读一个 tile 到共享内存 | 写「tile 累加」小例子（不算完整 GEMM），体会 tile 复用 | 理解「从全局读一次、共享内存多次复用」的价值 |
| Day 6 | 同步与竞态 | `__syncthreads` 的正确位置、`__syncthreads` 在分支内导致的死锁、读写 barrier | 故意在 `if` 分支里放 `__syncthreads`，观察错误/挂起 | 能写出不会死锁的同步逻辑 |
| Day 7 | 复习 + 综合 | 转置三版本（全局直接 / 共享内存 / 共享内存+padding）统一 benchmark | 完成三版本性能对比，记录数字与结论 | 交出「转置优化对比」笔记（博客素材） |

**Week 1 自检：** 能画出 Bank Conflict 示意图；能手写 `__shared__` + `__syncthreads` 的转置 kernel；能解释 padding 原理。

---

## Week 2：并行归约（Reduction）

**周目标：** 实现「求数组和/最大值」的高性能归约，从全局内存版一路优化到 warp shuffle + 多 block。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 归约概念 | 归约 = 把 N 个元素合并成 1 个（sum/max/min）；树形归约 vs 串行；`log N` 步 | 写串行归约 + 画树形归约图 | 能解释归约的并行度如何随层级递减 |
| Day 2 | Naive 归约 | 每个 block 一个线程顺序加、或全局 `atomicAdd` 竞争 | 写全局内存 + `atomicAdd` 版归约 | 跑通，记录极慢的基线带宽 |
| Day 3 | 共享内存归约 | block 内树形归约：每轮步长减半、`__syncthreads` 分隔 | 写 shared-memory 树形归约，避免 bank conflict（stride 选择） | 对拍通过，理解 stride 与 conflict 的关系 |
| Day 4 | Warp Shuffle 归约 | `__shfl_down_sync`/`__shfl_xor_sync`、`0xffffffff` 掩码、免同步 | 写 warp shuffle 归约（block 内先 warp 内归约再跨 warp） | 比 shared-memory 版更快，能解释为何免 `__syncthreads` |
| Day 5 | 多 block 归约 | 单 block 结果只是部分和；两种做法：`atomicAdd` 或第二次 kernel | 写「每 block 得部分和 → atomicAdd 汇总」完整版 | 大数组归约通过 CPU 对拍 |
| Day 6 | 优化技巧 | 模板参数指定 block size、`#pragma unroll`、`volatile`/`__syncthreads` 时机、首轮合并 | 把 block size 做成模板参数，测 128/256/512 性能 | 记录各配置的性能差异，形成结论 |
| Day 7 | 复习 + 综合 | 各版本（naive/shared/shuffle/multi-block）统一 benchmark，画性能曲线 | 完成「归约进化史」对比 + 笔记 | 交出归约优化对比（博客素材） |

**Week 2 自检：** 能手写 `__shfl_down_sync` 归约；能解释 shared-memory 归约 vs shuffle 归约的差异；能写多 block 归约并汇总。

---

## Week 3：前缀和（Scan）

**周目标：** 实现 inclusive/exclusive 前缀和，理解 Hillis-Steele 与 Blelloch 两种经典算法，以及 scan 在流压缩中的应用。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | Scan 概念 | 前缀和 `out[i]=Σin[0..i]`；inclusive vs exclusive；scan 是很多算法的地基（排序/压缩/直方图） | 写串行前缀和，明确 inclusive/exclusive 差异 | 能区分两种定义并写出递推 |
| Day 2 | Hillis-Steele Scan | 每轮跨 1、2、4… 步相加，`log N` 轮、`N` 次工作；适合简单实现 | 写 Hillis-Steele 版（含 `__syncthreads`），对拍 | 跑通，理解「倍增步长」思想 |
| Day 3 | Blelloch Scan（上扫） | up-sweep：树形两两相加，得到总量与中间部分和 | 写 up-sweep 阶段 | 能画出 up-sweep 的树形过程 |
| Day 4 | Blelloch Scan（下扫） | down-sweep：从根往下分发，得到完整前缀和；work-efficient | 完成 down-sweep，拼出完整 Blelloch scan，对拍 | 理解 Blelloch 是 work-efficient（O(N) 工作） |
| Day 5 | 多 block Scan | 先 block 内 scan，再 block 间 scan 得 offset，再加回 | 写「block 内 scan + 全局 offset」两阶段版 | 大数组 scan 通过 CPU 对拍 |
| Day 6 | Scan 应用：流压缩 | stream compaction：用 `1/0` 掩码保留元素；先 scan 掩码再 scatter | 写流压缩（保留满足条件的元素），对拍 | 能解释 scan 如何用于压缩 |
| Day 7 | 复习 + 综合 | 完整 scan（单 block + 多 block）+ 流压缩 + benchmark | 完成 scan 完整实现 + 对拍 + 笔记 | 交出 scan 实现（P1 三件套之一完成） |

**Week 3 自检：** 能手写 Hillis-Steele 或 Blelloch scan；能写多 block scan 并加 offset；能用一个 scan 应用（流压缩）讲清 scan 的价值。

---

## Week 4：向量化、Occupancy 与 P1 收尾

**周目标：** 掌握向量化访存与 Occupancy 调优，把前 3 周成果整合成 P1 基础算子库。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 向量化访存 | `float4`/`int4` 一次读 16 字节；内存对齐（`__align__`/`cudaMalloc` 已对齐）；`reinterpret_cast` | 把 element-wise 改成 `float4` 版，对比带宽 | 理解向量化为何提升访存吞吐 |
| Day 2 | 循环展开与 ILP | `#pragma unroll`、手动展开、指令级并行（多个独立访存在飞） | 对比展开前后 kernel 的带宽与寄存器数 | 能解释「更多在飞的访存 → 更好隐藏延迟」 |
| Day 3 | Occupancy 深入 | occupancy = 活跃 warp / SM 最大 warp；受寄存器、共享内存限制；`--ptxas-options=-v` 看资源 | 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算 occupancy | 能根据寄存器/共享内存算出 occupancy |
| Day 4 | 资源权衡 | 更多共享内存 vs 更高 occupancy 的矛盾；不同 kernel 的最优点不同 | 对归约/转置改 tile 大小，画「性能 vs tile」曲线 | 能解释「越大 tile 不一定越快」 |
| Day 5 | P1 整合 | 统一接口 `template<class T> void reduce(const T* in, T* out, int n)`；统一对拍 + 计时 + 错误检查 | 把 element-wise/reduction/transpose/scan 收进一个库，写统一测试入口 | 四个算子接口统一、对拍通过 |
| Day 6 | Benchmark 脚本 | 脚本化跑不同尺寸、输出 CSV、用 matplotlib 画性能曲线 | 写 benchmark 脚本，产出带宽/时间曲线图 | 交出可复现的 benchmark + 曲线图 |
| Day 7 | 月度复习 + 里程碑 | 复盘 M2；P1 打 tag；写第 1 篇博客《手写 CUDA 归约/转置优化》 | 完成月度自检 + 发布博客 + P1 仓库 README | 交出 P1 算子库 + 第 1 篇博客 |

**M2 月度自检清单：**
- [ ] 能画出并解释 Bank Conflict，会用 padding 消除。
- [ ] 手写 shared-memory 归约、warp shuffle 归约、多 block 归约，对拍通过。
- [ ] 手写 Hillis-Steele / Blelloch scan + 多 block offset，对拍通过。
- [ ] 会用 float4 向量化与 `#pragma unroll`，能算 occupancy。
- [ ] P1 算子库完成（4 类算子 + 对拍 + benchmark + 性能曲线）。
- [ ] 第 1 篇技术博客已发布。

## 本月背书行动（在职）

- **发第 1 篇博客**：把「归约/转置优化对比」写成深度文（原理 + 代码 + 性能数字），发知乎/公众号/个人博客。
- **P1 打 tag + benchmark**：仓库打 `v1.0`，README 附可复现的带宽/加速比表格。
- 产出口径：**1 篇博客 + P1 项目（含 benchmark 数字）**。

## Month 2 参考资源

- PMPP 第 6–9 章（性能、归约、scan、共享内存）
- CUDA C++ Best Practices Guide：合并访问、共享内存、occupancy 章节
- NVIDIA 博客《Optimizing Parallel Reduction in CUDA》
- CUDA 官方 `samples/6_Performance/reduction`、`transpose` 示例
