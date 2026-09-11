# compiler_lab/

编译器与 lowering 链实验（对应 M5）。

| 子目录 | 内容 |
|---|---|
| `toy_ir/` | 手写 Python toy IR + elementwise fusion pass + tiling metadata/codegen（mini-compiler v0） |
| `torch_compile/` | Dynamo / FX / AOTAutograd / Inductor 实验、graph break 修复、TTIR/TTGIR/PTX dump |

目标：建立「Python graph → FX → Triton IR → PTX → SASS」的完整 lowering 认知，并能用 toy IR 写一个 fusion pass。
