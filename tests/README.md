# tests/

正确性与回归测试。

- 每个算子：CPU/PyTorch reference 对拍；低精度明确 `atol/rtol`；训练算子补 `gradcheck`。
- runtime 组件：request state machine、block allocator 等单元测试。
- 目标：一条命令跑全部测试，CI 可接入（M1 Stretch 目标）。
