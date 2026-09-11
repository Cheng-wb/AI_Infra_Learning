# operators/

面向框架接入的高层算子实现，按算子类型分类。这里的算子应能被封装成 PyTorch custom op 并兼容 `torch.compile`（见 M5）。

| 子目录 | 内容 | 主要月份 |
|---|---|---|
| `gemm/` | naive → tiled → register-tiled → Tensor Core GEMM | M2、M3 |
| `normalization/` | RMSNorm、LayerNorm、Softmax（stable/online） | M4 |
| `attention/` | MHA/GQA/MQA、FlashAttention、PagedAttention | M4、M6 |
| `quantization/` | INT8/FP8/FP4 quant/dequant、low-precision GEMM | M7 |
| `moe/` | top-k routing、permute、grouped GEMM、fused experts | M8 |

每个算子提供：CPU/PyTorch reference → GPU 实现 → 对拍 → benchmark → 文档。
