# 模型扩展

模型继承 `SERModel`，并实现三个边界：

- `model_spec`：输入 key、`TensorSpec`、mask/变长支持和类别数；
- `model_config`：可 JSON 序列化且足以从注册表重建的完整配置；
- `forward(SERBatch)`：返回 logits 为 `[B,C]` 的 `ModelOutput`。

配置使用 `StrictConfig`，注册时提供 `ModelDescriptor` 和配置 schema。模型不能
自行读取音频、解析 manifest 或猜测输入 layout。变长模型必须使用 lengths/mask
排除 padding；测试需覆盖配置往返、padding 隔离、训练一步和 artifact 往返。

第三方适配器不得执行远程代码。若构造依赖外部模型目录，导出前应把重建架构所需
配置固化进 `model_config`，确保 artifact 可离线恢复。
