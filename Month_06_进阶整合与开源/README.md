# Month 6：进阶整合、量化入门与开源起航

> **主题：** 把前 5 个月的算子整合成完整的 Capstone v1；入门量化（INT8/FP8/BF16）与 kernel fusion；正式进入开源贡献。
> **前置：** 完成 M5（算子可接 PyTorch、会 Nsight）。
> **月度目标：** 完成 **P5 Capstone v1 算子库**；理解并实现量化与融合算子；争取 **1–2 个 merged PR**；达成「可投第一份 AI Infra / 算子岗」的 checkpoint。

## 月度里程碑

- ✅ 理解 INT8/FP8/BF16 的表示、量化/反量化、对算子结果的影响。
- ✅ 掌握 kernel fusion（如 fused Layernorm+GEMM、fused bias+activation），减少访存。
- ✅ 完成 P5：一个完整开源算子库（含融合算子 + 量化 + benchmark + 文档）。
- ✅ 完成首个开源 merged PR，掌握「issue → PR → review → merge」全流程。
- ✅ 达成 6 个月 checkpoint 自检。

---

## Week 1：量化与数值精度

**周目标：** 理解低精度计算的表示、量化方法与精度影响，为算子支持 INT8/FP8 打基础。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 数值表示 | FP32/BF16/FP16/FP8/INT8 的位宽、范围、精度、舍入 | 手写一个「打印各类型能表示的边界」小工具 | 能说清 BF16 与 FP16 的区别（范围 vs 精度） |
| Day 2 | 量化原理 | 对称/非对称量化、scale/zero-point、量化→反量化、per-tensor/per-channel | 手写 CPU 量化/反量化，对拍 | 能写出 `q = round(x/scale) + zp` |
| Day 3 | INT8 量化算子 | 把 GEMM 改成 INT8 输入 + INT32 累加（可用 WMMA INT8） | 写 INT8 GEMM 或量化 matmul，对拍 | 跑通，理解 INT32 累加 |
| Day 4 | FP8 简介 | E4M3/E5M2 两种 FP8、缩放因子、NVIDIA FP8 路线（M9 深挖） | 读 NVIDIA FP8 白皮书/博客 | 理解 FP8 与 INT8 的权衡 |
| Day 5 | 量化误差分析 | 量化误差来源、对模型精度的影响、校准（calibration） | 对比 FP32 vs INT8 算子的输出误差分布 | 能说清量化精度损失 |
| Day 6 | 量化在 LLM 的应用 | GPTQ/AWQ/FP8-LLM 思路概览（M9 深挖） | 读一篇量化综述 | 理解「权重量化 vs 激活量化」 |
| Day 7 | 复习 + 综合 | 整理量化笔记 + 量化算子 | 完成 Week 1 笔记 | 交出量化笔记 |

**Week 1 自检：** 能说清各精度类型的表示与权衡；能手写量化/反量化；能写 INT8 算子并分析误差。

---

## Week 2：Kernel Fusion 与稀疏

**周目标：** 掌握算子融合的收益（减少访存），实现融合算子，了解 2:4 稀疏。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 融合的动机 | 逐算子执行有多次读写显存；融合成 1 个 kernel 省掉中间结果往返 | 用 nsight 看「LN + GEMM 两算子」的访存 | 能说清融合省了哪些访存 |
| Day 2 | fused bias + activation | GEMM 后接 bias 和 relu，融合进 epilogue | 写「GEMM + bias + relu」融合算子 | 跑通，对比分开执行 |
| Day 3 | fused LayerNorm + GEMM | LN 的输出直接喂给 GEMM，不再写回显存 | 写 fused LN+GEMM（或 LN+Linear） | 融合版显存/时间更优 |
| Day 4 | fused 反向 | 融合算子的反向也要一起实现，保存必要中间量 | 写 fused 算子的 backward | gradcheck 通过 |
| Day 5 | 2:4 稀疏 | 结构化稀疏（每 4 个元素 2 个非零），Tensor Core 支持 | 读稀疏 Tensor Core 资料，实现稀疏 mask | 理解稀疏加速原理 |
| Day 6 | 稀疏算子 | 稀疏 GEMM 的思路（压缩存储 + 稀疏 mma） | 实现一个简单稀疏 matmul 雏形 | 能说清稀疏存储格式 |
| Day 7 | 复习 + 综合 | 整理融合/稀疏笔记 | 完成 Week 2 笔记 | 交出融合/稀疏笔记 |

**Week 2 自检：** 能说清融合的访存收益；能手写 fused LN+GEMM（含反向）；理解 2:4 稀疏。

---

## Week 3：Capstone v1 整合（P5）

**周目标：** 把前 6 个月的所有算子整合成一个完整、有文档、可 benchmark 的开源算子库。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 库架构设计 | 目录结构、统一接口、命名规范、`CMakeLists` + `setup.py` 双构建 | 设计 P5 库结构并初始化 | 有清晰架构 |
| Day 2 | 算子汇总 | element-wise/reduction/transpose/scan/GEMM/softmax/LN/attention/量化 全部入册 | 把所有算子收进统一接口 | 算子清单完整 |
| Day 3 | 统一测试 | 每个算子配 CPU/PyTorch 对拍 + gradcheck | 写统一测试入口 | 全算子测试通过 |
| Day 4 | 统一 benchmark | 脚本化跑所有算子，输出 CSV + 曲线，对比 cuBLAS/cuDNN/PyTorch | 完成 benchmark 脚本 | 有可复现数字 |
| Day 5 | 文档与 README | 写清：功能、用法、性能表、环境、如何复现 | 写完整 README + 性能表 | README 达到「陌生人能复现」标准 |
| Day 6 | 发布 | 打 tag `v1.0`、写 release note、分享到社区 | 发布仓库 | 仓库公开可用 |
| Day 7 | 复盘 | 复盘 6 个月，写阶段总结 + 做 checkpoint 自检 | 完成月度自检 + 阶段总结 | 交出 P5 + 阶段总结 |

**Week 3 自检：** P5 算子库完整、可复现、有 benchmark、有文档、已发布。

---

## Week 4：开源贡献实战 + Checkpoint

**周目标：** 完成开源贡献闭环（issue → PR → review → merge），达成 6 个月 checkpoint。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 开源工作流 | fork、branch、规范 commit、PR 描述、响应 review、CI 通过 | 复习 git 协作流程 | 能规范走完 PR 流程 |
| Day 2 | 精修 M4/M5 的 PR | 回应 reviewer 意见，修改代码，推 CI | 让已提交的 PR 进入可合并状态 | PR 通过 CI |
| Day 3 | 找新 issue | 在 PyTorch/Triton/vLLM 找第二个 `good first issue` 或算子 bug | 认领并分析第二个 issue | 第二份贡献启动 |
| Day 4 | 写新 patch | 完成第二个 issue 的代码 | 提交第二个 PR | 第二个 PR 提交 |
| Day 5 | checkpoint 自检 | 对照 README 第九节「第 6 个月 checkpoint」逐项自测 | 完成全部 checkpoint 检查项 | 逐项通过或明确差距 |
| Day 6 | 简历初稿 | 把 6 个月成果写成简历：项目 + 开源 + 博客 + 数字 | 写简历初稿 + 项目描述 | 有一版可投简历 |
| Day 7 | 阶段复盘 + 规划 | 复盘 Phase 1，规划 Phase 2（M7–M12）细节 | 写阶段总结 + 下阶段计划 | 交出 Phase 1 总结 |

**M6 月度自检 = 6 个月 checkpoint 清单：**
- [ ] 不看模板能写出正确、可编译、带错误检查的 CUDA kernel + CPU 对拍。
- [ ] 能说清算子的算术强度、Roofline 判断瓶颈、给出 ≥2 种优化手段。
- [ ] 手写 GEMM 达到 cuBLAS ≥50%（理想 ≥80%）。
- [ ] 算子封装成 PyTorch 自定义算子（含反向），可训练。
- [ ] 能用 ncu 定位并消除真实瓶颈。
- [ ] P5 完整开源算子库已发布（含量化 + 融合算子）。
- [ ] 1–2 个 merged PR + ≥4 篇博客 + 简历初稿。

## 本月背书行动（在职）

- **争取 1–2 个 merged PR**：精修 M4/M5 提交的 PR 到 merge，再提交第二个。
- **P5 打 tag 发布** + 完整 benchmark。
- **简历初稿**：把 6 个月成果（项目 + PR + 博客 + 数字）写成可投简历。
- 产出口径：**1–2 merged PR + P5 发布 + 简历初稿**，达成「可投第一份 AI Infra 岗」。

## Month 6 参考资源

- NVIDIA FP8 / INT8 官方博客与白皮书
- GPTQ / AWQ / FP8-LLM 论文（概念级）
- CUTLASS 文档的 epilogue / 稀疏章节
- 目标开源仓库的 `CONTRIBUTING.md`（PyTorch/Triton/vLLM）
