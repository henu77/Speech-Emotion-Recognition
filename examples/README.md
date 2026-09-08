# 可执行示例

示例只调用公开 API，可从仓库根目录运行：

```bash
python examples/train_from_python.py configs/cnn_logmel.yaml
python examples/predict_artifact.py artifacts/model path/to/audio.wav
```

训练示例要求配置所引用的标准 manifest 已存在。推理示例要求先从可信 checkpoint
导出 artifact。完整 CLI 工作流见 `docs/TRAINING_AND_CLI.md`。
