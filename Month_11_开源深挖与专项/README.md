# Month 11：开源深挖、专项强化与 Capstone v2

> **月目标：** 从「做自己的 demo」升级到「能在真实大型项目里定位、修改、验证」。
> **Must：** 深读一个目标 codebase；建立 benchmark reproducer；提交有技术证据的 PR。
> **Should：** 明确一个主 specialization。
> **Stretch：** 做出性能类 merged PR 或被 maintainer 深度 review 的高质量 PR。

### 本月先选择主方向

- **Kernel Track：** CUTLASS / CuTe DSL / FlashInfer / Triton kernels。
- **Compiler Track：** PyTorch Inductor / Triton compiler / MLIR/LLVM 相关路径。
- **Runtime Track：** vLLM / SGLang scheduler、KV cache、distributed execution。

不需要把三条都学到同一深度；M1–M10 已经建立广度，M11 开始形成真正的 T 型能力。

## Week 1 — Codebase Deep Dive

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Build/test | 从源码构建目标项目 | 测试可跑 |
| 2 | Architecture | 画模块/数据流/关键 abstraction | 1 页架构图 |
| 3 | Hot path | 跟踪一次真实调用 | 找到 kernel/compiler/runtime 核心路径 |
| 4 | Benchmark suite | 学官方 benchmark 与 CI | 能复现一个基准 |
| 5 | Recent PRs | 阅读 3–5 个近期性能 PR | 总结 maintainer 标准 |
| 6 | Issue triage | 选 2–3 个可贡献问题 | 写 feasibility note |
| 7 | 周收口 | 锁定一个主 issue | 有 reproducible baseline |

## Week 2 — Reproducer 与 Root Cause

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Minimal repro | 把问题缩成最小 workload | 稳定复现 |
| 2 | Measurement | 固定环境/shape/dtype/seed | 数据稳定 |
| 3 | Profiler | ncu/nsys/compiler dump | 有根因证据 |
| 4 | Hypothesis A | 做第一种修改 | 结果记录 |
| 5 | Hypothesis B | 第二种修改/排除法 | 根因收敛 |
| 6 | Correctness | edge cases/regression tests | 不以性能换错误 |
| 7 | 周收口 | issue analysis 文档 | maintainer 能读懂 |

## Week 3 — Patch / PR

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Implementation | 写 production-quality patch | style/build 通过 |
| 2 | Tests | 加单测/回归测试 | 覆盖 bug/性能条件 |
| 3 | Benchmark | 多 shape / arch / dtype | 没有 cherry-pick 数据 |
| 4 | PR description | 背景、根因、修改、风险、数字 | 描述完整 |
| 5 | Submit | 提 PR | CI 运行 |
| 6 | Review | 响应 review | 每个反馈有技术回应 |
| 7 | 周收口 | 整理贡献笔记 | merge 不受控，质量可控 |

## Week 4 — Capstone v2

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Capstone scope | 把 M4/M6/M9/M10 组合成一条故事线 | 明确问题/指标 |
| 2 | Kernel layer | 用最佳 kernel/DSL 路径 | benchmark |
| 3 | Runtime layer | cache/scheduler/graph 集成 | end-to-end 跑通 |
| 4 | Multi-GPU layer | 能做则加入 TP/EP；否则写可扩展设计 | scaling 或设计证据 |
| 5 | Profiling | 端到端 bottleneck analysis | top bottleneck 明确 |
| 6 | Documentation | 架构图、复现、数字、trade-off | portfolio-ready |
| 7 | 月收口 | Capstone v2 release candidate | 作为简历第一项目 |

## Month 11 自检

- [ ] 从源码构建并深读一个目标 codebase，画 1 页架构图，锁定并复现一个主 issue。
- [ ] 有最小 reproducer + 根因证据（ncu/nsys/compiler dump），issue analysis 能被 maintainer 读懂。
- [ ] 提交 1 个深度 PR（production-quality patch + 单测 + 多 shape/arch/dtype benchmark + 完整描述）。
- [ ] 明确主 specialization（Kernel / Compiler / Runtime 三选一）。
- [ ] Capstone v2 release candidate 完成（kernel + runtime，最好含 distributed），作为简历第一项目。

> 完成后进入 [Month 12 — Capstone / 面试 / 正式投递](../Month_12_Capstone与面试/README.md)。
