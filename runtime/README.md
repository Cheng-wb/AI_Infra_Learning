# runtime/

LLM 推理运行时组件（对应 M6、M10）。

| 子目录 | 内容 |
|---|---|
| `kv_cache/` | contiguous KV cache manager、capacity 计算器 |
| `paged_attention/` | block allocator、block table、paged KV read/write、fragmentation 分析 |
| `scheduler/` | request state machine、continuous batching、token budget、chunked prefill、prefix caching、eviction |
| `mini_server/` | CUDA Graph decode、scheduler + cache + attention 集成的 toy generation loop |

目标：跑通一个 Mini LLM Inference Runtime v1，并能用 TTFT/TPOT/throughput/memory 出具可复现报告。
