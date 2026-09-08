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

## 数据集导入器

`ser dataset scan` 只预览并报告问题，`ser dataset import` 在扫描无错误后写入标准
manifest。内置 importer 如下：

| ID | 输入 | 默认划分重点 |
|---|---|---|
| `folder` | 标签目录中的音频 | 可配置比例 |
| `csv` / `jsonl` | 外部表格或逐行记录 | 保留来源 split |
| `casia` | CASIA 说话人/情感目录 | 说话人独立 |
| `ravdess` | 官方七段文件名 | 解析演员、强度和语音/歌曲通道 |
| `csemotions` | metadata CSV + `wav_data` | 性别平衡、说话人独立 |
| `esd` | 20 位说话人目录与转写 | 语言分层、说话人独立 |
| `crema_d` | `AudioWAV` + 人口统计 CSV | 性别分层、演员独立 |
| `emotiontalk` | 逐句 JSON/WAV | 说话人独立或官方对话划分 |

专用目录结构、标签语义、许可提醒和命令示例见
[`data/README.md`](../data/README.md)。Importer 默认只引用原始音频，不复制、移动或
重新分发数据。自定义 split 必须覆盖全部说话人且不得重复。
