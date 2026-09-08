# 公共 API 索引

稳定入口优先从对应子包导入，顶层 `ser_lib` 只暴露最常用类型。

| 子包 | 主要公开能力 |
|---|---|
| `ser_lib.core` | 严格配置、事件、取消、日志和根异常 |
| `ser_lib.data` | Manifest、AudioLoader、SERDataset、Pipeline、Collator、注册表与 profile |
| `ser_lib.models` | SERModel、ModelOutput、CNN、GRU、Transformer、HF adapter 与注册表 |
| `ser_lib.engine` | ExperimentConfig、Trainer、Evaluator、optimizer/scheduler 和 checkpoint |
| `ser_lib.artifacts` | artifact 导出、验证、加载和模型卡 |
| `ser_lib.inference` | 单文件、批量和纯 PCM 流式推理 |
| `ser_lib.benchmark` | 可序列化微基准和同环境回归比较 |
| `ser_lib.cli` | `ser` 命令入口 |

公开模型和组件应通过 registry descriptor 发现，不要依赖内部字典。未列入各模块
`__all__` 的名称视为实现细节，可能在小版本中调整。当前版本为 `0.2.0`，尚未
承诺 1.0 级别的长期兼容性。

## 类别不平衡与训练目标

- `ser_lib.engine.LossConfig`：交叉熵、focal loss、类别权重和 label smoothing。
- `ser_lib.engine.SamplingConfig`：随机打乱或 WeightedRandomSampler 配置。
- `ser_lib.engine.ClassificationLoss`：经过严格配置校验的分类损失。
- `ser_lib.engine.build_weighted_sampler`：根据训练标签构建确定性平衡采样器。

## 数据导入器

内置 importer 从 `ser_lib.data.importers` 导入，包括 `FolderImporter`、
`CsvImporter`、`JsonlImporter`、`CasiaImporter`、`RavdessImporter`、
`CsemotionsImporter`、`EsdImporter`、`CremaDImporter` 和 `EmotionTalkImporter`。
所有实现遵循 `scan(source, config) -> ImportPreview` 与
`convert(source, destination, config) -> DatasetManifest` 契约；组件发现应优先使用
registry descriptor，而不是硬编码类列表。
