# 实验配置模板

这些配置展示 `ExperimentConfig` 的完整结构。默认假设已经通过 `ser dataset import`
在 `data/standard/dataset.yaml` 创建了六分类数据集；使用前必须根据自己的 manifest
修改 `data.manifest`、`data.labels` 和模型的 `num_classes`。

```bash
ser train configs/cnn_logmel.yaml --split train --batch-size 16
ser train configs/gru_mfcc.yaml --split train --batch-size 16
ser train configs/transformer_logmel.yaml --split train --batch-size 16
ser train configs/csemotions_cnn_logmel.yaml --split train --batch-size 32
ser train configs/esd_cnn_logmel.yaml --split train --batch-size 32
ser train configs/crema_d_cnn_logmel.yaml --split train --batch-size 32
ser train configs/emotiontalk_cnn_logmel.yaml --split train --batch-size 32
```

配置中的相对路径始终相对于配置文件自身，而不是当前工作目录。CNN 与 Transformer
模板使用 64 维 Log-Mel，GRU 模板使用 40 维 MFCC；表示的维度必须与模型的
`feature_dim` 完全一致。

`csemotions_cnn_logmel.yaml` 是本地 CSEMOTIONS 专用配置，需先运行
`ser dataset import --importer csemotions ...` 生成标准 manifest。

`esd_cnn_logmel.yaml` 是中英双语 ESD 五分类配置，需先运行
`ser dataset import --importer esd ...` 生成语言分层、说话人独立的标准 manifest。

`crema_d_cnn_logmel.yaml` 使用 CREMA-D 文件名中的六类表演情感标签，需先运行
`ser dataset import --importer crema_d ...` 生成性别分层、演员互斥的标准 manifest。

`emotiontalk_cnn_logmel.yaml` 是中文 EmotionTalk 七分类配置。由于标签不均衡，默认
组合 focal loss 与 WeightedRandomSampler；需先通过 `emotiontalk` importer 生成 manifest。
