# Month 5：PyTorch Custom Op、torch.compile 与 Compiler 基础

> **月目标：** 把「独立 kernel」变成「框架可组合算子」，并建立从 Python graph 到 GPU code 的编译链认知。
> **Must：** `torch.library`、custom op、opcheck、autograd、`torch.compile`。
> **Should：** Inductor/FX/AOTAutograd、Triton IR。
> **Stretch：** 做一个极简 graph compiler / fusion pass。

## Week 1 — Modern PyTorch Custom Operators

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Custom op API | 用 `torch.library.custom_op` 封装一个 CUDA/Triton op | eager 正确 |
| 2 | Schema / mutation / alias | 给 op 定义规范 schema | opcheck schema 通过 |
| 3 | Fake/meta kernel | 添加 FakeTensor/meta 行为 | dynamic shape 测试 |
| 4 | Autograd registration | 用 `register_autograd` 注册 backward | gradcheck 通过 |
| 5 | Triton op | 用 `torch.library.triton_op` 包装 Triton kernel | `torch.compile` 可 trace |
| 6 | C++ TORCH_LIBRARY | 写一个 C++/CUDA registration 版本 | Python 侧可调用 |
| 7 | 周收口 | 形成 custom-op 模板仓库 | opcheck + gradcheck + tests |

## Week 2 — torch.compile / Dynamo / AOTAutograd / Inductor

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | Dynamo | compile 一个小 Transformer block | 能解释 graph capture |
| 2 | Graph break | 人为制造 graph break 并修复 | 记录原因和解决方式 |
| 3 | FX Graph | 导出/打印 graph，识别 op pattern | 能读基本 FX IR |
| 4 | AOTAutograd | 观察 forward/backward graph | 能解释训练编译链 |
| 5 | Inductor | 查看生成 Triton/C++ 代码 | 找到 fusion/codegen 结果 |
| 6 | Custom op + compile | 把 M4 kernel 放入 compiled model | eager/compiled 都正确 |
| 7 | 周收口 | 写 PyTorch compile pipeline 图 | Dynamo→AOT→Inductor 清楚 |

## Week 3 — IR、PTX、SASS 与 Compiler Thinking

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | IR 层次 | 对同一 op 收集 FX/Triton IR/PTX/SASS | 画 lowering chain |
| 2 | PTX | 识别 load/store/fma/barrier/mma | 能把指令映射回源码 |
| 3 | SASS | 对比 naive/tiled 两版 | 用指令/资源解释差异 |
| 4 | MLIR 概念 | 学 dialect/op/type/region/pass | 能读一个简单 MLIR snippet |
| 5 | Rewrite/Fusion | 手写 Python toy IR + elementwise fusion | pass 有单测 |
| 6 | Tiling pass | toy matmul IR 加 tile metadata/codegen | 生成可运行 Triton/CUDA skeleton |
| 7 | 周收口 | mini-compiler v0 | 输入 graph→优化→生成代码 |

## Week 4 — P5 Compile-friendly Operator Library

| Day | 主题 | 动手任务 | 验收 |
|---|---|---|---|
| 1 | 选 3 个算子 | RMSNorm/RoPE/fused activation 等接入 | API 统一 |
| 2 | Dynamic shapes | 测不同 batch/seq/head dim | 不写死单 shape |
| 3 | torch.compile | compiled Transformer layer | 无意外 graph break |
| 4 | AOT/部署认知 | 体验 AOTInductor 或 export 路径 | 写部署说明 |
| 5 | Performance | eager vs compile vs custom kernel | 同口径 benchmark |
| 6 | 文档/CI | 测试矩阵 + CI + compatibility note | 工程可复现 |
| 7 | 月收口 | 发布 P5 + compiler 博客 | 能讲「kernel 如何进入 framework」 |

## Month 5 自检

- [ ] 用 `torch.library.custom_op` / `triton_op` 注册 custom op，opcheck + gradcheck + dynamic shape 通过。
- [ ] 能解释 Dynamo → AOTAutograd → Inductor 链路，读 FX IR，修复 graph break。
- [ ] 能收集同一 op 的 FX/Triton IR/PTX/SASS 并画 lowering chain。
- [ ] 手写 toy IR + elementwise fusion pass（有单测），mini-compiler v0 能 graph→优化→生成代码。
- [ ] P5 compile-friendly custom-op library + mini compiler 已发布（eager vs compile vs custom kernel 同口径 benchmark）。

> 完成后进入 [Month 6 — KV Cache / PagedAttention / Scheduler](../Month_06_推理运行时与KV调度/README.md)。
