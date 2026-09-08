# Artifact 与安全边界

schema v2 artifact 包含 safetensors 权重、数据/模型配置、标签、指标、模型卡和
manifest。manifest 保存所有组成文件的 SHA-256；加载器先验证路径、版本、文件
存在性、哈希及外部元数据一致性，再构建模型。

```bash
ser artifact export --config configs/cnn_logmel.yaml \
  --checkpoint runs/cnn-logmel/checkpoints/best.pt --destination artifacts/model
ser artifact verify artifacts/model --json
ser artifact inspect artifacts/model --json
```

导出目标必须不存在，防止覆盖已有模型。artifact 是目录级原子写入。旧 v1
PyTorch 权重只有在调用方明确授权可信 pickle 时才允许加载。发布模型时必须补充
模型卡中的训练数据、语言、许可、用途和限制，不得把 checkpoint 冒充 artifact。
