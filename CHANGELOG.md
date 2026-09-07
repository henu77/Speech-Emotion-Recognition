# Changelog

本项目遵循语义化版本。当前内容尚未发布。

## [Unreleased]

### Added

- 统一 Manifest、Representation、Collator 数据流水线。
- CNN、GRU、轻量 Transformer 和可选 Hugging Face 编码器模型。
- 可复现训练、完整 SER 指标、checkpoint v2 和安全 artifact v2。
- 单文件、批量、纯 PCM 流式推理与 `ser` CLI。
- RAVDESS importer、音频 profile 和 benchmark 回归比较。

### Changed

- 完全移除三个旧 Dataset 包装器，统一使用 `SERDataset`。
- 仓库边界收缩为纯 SER 基础库。

### Removed

- 桌面端、Web UI 和本地 HTTP service 代码。
