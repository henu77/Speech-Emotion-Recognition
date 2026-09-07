# 训练与 CLI

训练以版本化 `ExperimentConfig` 为唯一配置源，组合 data、model、trainer、
optimizer、scheduler 和 output_dir。相对路径按配置文件位置解析。

```bash
ser train configs/experiment.yaml --split train --batch-size 16
ser train configs/experiment.yaml --resume runs/checkpoints/epoch-0005.pt
ser evaluate artifacts/model --manifest data/dataset.yaml --split test --output runs/eval
ser predict artifacts/model path/to/audio --output runs/predictions.jsonl
```

未配置 checkpoint 目录时 CLI 使用 `output_dir/checkpoints`。checkpoint 用于可信
本地续训，不用于分发；评估和推理以 artifact 为入口。所有命令支持 `--help`，
主要结果可通过 `--json` 提供给脚本调用。
