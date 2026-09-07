# 标准数据格式

数据集由一个 `dataset.yaml` 和一个或多个 JSONL split 构成：

```yaml
schema_version: 1
dataset_id: demo
root: ./audio
splits:
  train: train.jsonl
labels:
  0: {en: neutral, zh: 平静}
  1: {en: happy, zh: 高兴}
```

每行 JSONL 至少包含唯一 `uid` 与 `audio_path`；训练数据还应包含从 0 开始的
连续整数 `label`。可选字段包括 `start_ms`、`end_ms`、`speaker_id`、
`sample_rate_hint` 和 `metadata`。

路径解析不依赖当前工作目录：split 相对 `dataset.yaml`，音频相对 `root`。
Waveform、Mel、MFCC 等差异属于 Representation，不属于 Dataset 类型。

```bash
ser dataset validate path/to/dataset.yaml --check-files --json
ser dataset stats path/to/dataset.yaml --probe-audio --json
```
