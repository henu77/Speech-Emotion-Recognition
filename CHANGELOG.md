# Changelog

本项目遵循语义化版本。当前内容尚未发布。

## [Unreleased]

### Added

- 统一 Manifest、Representation、Collator 数据流水线。
- CNN、GRU、轻量 Transformer 和可选 Hugging Face 编码器模型。
- 可复现训练、完整 SER 指标、checkpoint v2 和安全 artifact v2。
- 单文件、批量、纯 PCM 流式推理与 `ser` CLI。
- RAVDESS importer、音频 profile 和 benchmark 回归比较。
- CNN、GRU 和 Transformer 实验配置模板及可执行 Python API 示例。
- 覆盖真实 WAV 数据链路的单 epoch 训练冒烟测试，并接入跨平台 CI。
- 验证集训练、可配置监控指标、early stopping 及 best/last checkpoint。
- 类别加权交叉熵、focal loss、WeightedRandomSampler 和逐 epoch JSONL 日志。
- weighted precision/recall/F1、balanced accuracy、MCC 与 Cohen's kappa 指标。
- CSEMOTIONS 专用 importer、元数据保留和性别平衡的说话人独立划分。
- ESD 中英双语 importer、转写保留、语言过滤和语言分层的说话人独立划分。
- CREMA-D importer、人口统计元数据、固定语句解析和性别分层的演员独立划分。
- EmotionTalk importer、完整逐句标注保留及说话人独立/官方对话两种划分策略。
- CSEMOTIONS、ESD、CREMA-D 和 EmotionTalk 的可校验训练配置与模型卡。
- 真实 EmotionTalk 三轮 GPU 训练、artifact 导出和独立测试评估的工程验收记录。

### Changed

- 完全移除三个旧 Dataset 包装器，统一使用 `SERDataset`。
- 仓库边界收缩为纯 SER 基础库。

### Removed

- 桌面端、Web UI 和本地 HTTP service 代码。
