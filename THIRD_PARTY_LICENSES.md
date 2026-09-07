# 第三方许可清单

本文件记录直接运行时依赖的上游许可；发布前仍需根据锁定版本重新生成和审查
完整传递依赖清单。

| 依赖 | 用途 | 上游许可 |
|---|---|---|
| PyTorch | 张量、模型与训练 | BSD 风格许可及打包的第三方许可 |
| TorchAudio | 音频读取、重采样与特征 | BSD-2-Clause |
| Pydantic | 严格配置校验 | MIT |
| PyYAML | YAML 配置与 manifest | MIT |
| safetensors | 安全权重格式 | Apache-2.0 |
| Transformers（可选） | 预训练语音编码器适配 | Apache-2.0 |

数据集和预训练模型不因使用本库而获得重新许可。RAVDESS 等语料的许可提示见
[`data/readme.md`](data/readme.md)，artifact 发布者必须在模型卡中填写训练数据、
模型许可和限制。
