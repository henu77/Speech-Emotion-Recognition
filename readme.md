# Speech Emotion Recognition

[![CI](https://github.com/henu77/Speech-Emotion-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/henu77/Speech-Emotion-Recognition/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个面向工程实践的语音情感识别（Speech Emotion Recognition，SER）Python
基础库。项目提供从外部音频数据到标准数据集、模型训练、评估、模型分发和推理的
完整核心链路，不包含 Web 页面、桌面端 UI 或本地 HTTP 服务。

> 当前版本：**v0.2.0 发版候选**。仓库暂不附带数据集和预训练权重。

## v0.2.0 新增功能

- 统一数据流水线：使用一个 `SERDataset` 处理 Waveform、Spectrogram、Mel、
  Log-Mel、MFCC 及组合输入，不再按特征类型维护多个 Dataset。
- 标准数据集格式：使用 `dataset.yaml + JSONL manifest` 描述路径、标签、说话人、
  数据划分和音频片段。
- 数据导入与检查：支持 Folder、CSV、JSONL、CASIA、RAVDESS、CSEMOTIONS、ESD、
  CREMA-D 和 EmotionTalk，提供扫描、转换、校验、统计和音频属性分析。
- 可扩展组件体系：Importer、Representation、Transform 和 Model 均可通过注册表
  扩展，并在运行前检查输入布局、特征维度、类别数和批处理兼容性。
- 内置模型：提供 CNN、GRU/BiGRU 和轻量 Transformer 分类基线；可选接入
  Hugging Face 语音编码器。
- 训练与评估：支持确定性随机种子、CPU/单 GPU、AMP、梯度累积、梯度裁剪、
  学习率调度、断点续训、类别权重、focal loss、平衡采样和逐 epoch 日志。
- SER 指标：支持 loss、accuracy/WAR、UAR、macro/weighted-F1、weighted
  precision/recall、balanced accuracy、MCC、Cohen's kappa、逐类指标、混淆矩阵
  和样本级概率报告。
- 安全模型产物：artifact v2 使用 `safetensors`，保存模型、数据配置、标签、指标、
  模型卡和校验和；训练 checkpoint 与分发 artifact 明确分离。
- 完整推理链路：支持单文件、目录、文件列表、manifest 批量推理，以及纯 PCM
  流式窗口、静音过滤、概率平滑和背压控制。
- 命令行工具：提供组件发现、数据导入、训练、评估、推理和 artifact 管理命令。
- 工程质量：包含跨平台 CI、静态检查、类型检查、测试、构建验证和数据管线基准。

## 功能状态

| 模块 | 当前能力 | 状态 |
|---|---|---|
| 数据管线 | Manifest、音频加载、表示、增强、缓存、动态/定长/滑窗批处理 | 稳定 |
| 数据导入 | Folder、CSV、JSONL、CASIA、RAVDESS、CSEMOTIONS、ESD、CREMA-D、EmotionTalk | 稳定 |
| 模型 | CNN、GRU/BiGRU、轻量 Transformer | 稳定基线 |
| 预训练模型 | Hugging Face 音频分类适配器 | 可选依赖，首轮实现 |
| 训练 | CPU/单 GPU、AMP、类别权重/focal loss、平衡采样、验证、early stopping、训练日志 | 稳定 |
| 评估 | 常用 SER 分类指标和 JSON/JSONL 报告 | 稳定 |
| Artifact | safetensors、完整性校验、模型卡、可移植恢复 | 稳定 |
| 推理 | 单文件、批量和纯 PCM 流式推理 | 稳定核心 |
| CLI | dataset/train/evaluate/predict/artifact 命令 | 稳定核心 |
| 教程 | 0–7 章课程结构 | 提纲，尚不可执行 |

当前完整自动化测试基线为 `151 passed`。此外已在 RTX 4060 上使用真实 EmotionTalk
数据完成 3 epoch 训练、最佳 checkpoint 导出和独立测试集评估，证明混合采样率、
单双声道数据可贯通训练与 artifact 工作流；该短训结果仅用于工程验收，不代表正式基准。

“稳定”表示已有自动化测试覆盖的基础闭环，并不表示 API 已承诺永久不变。
在 1.0.0 之前，公共接口仍可能按照语义化版本规则调整。

## 安装

要求 Python 3.10–3.12。PyTorch 和 TorchAudio 应安装来自同一 CPU/CUDA 渠道的
兼容版本。

```bash
git clone https://github.com/henu77/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

需要 Hugging Face 预训练编码器时：

```bash
python -m pip install -e ".[pretrained]"
```

验证安装：

```bash
python -c "import ser_lib; print(ser_lib.__version__)"
ser components list --json
```

如果 `ser` 命令不可用，可使用 `python -m ser_lib.cli` 检查当前 Python 环境。

## 最短可运行示例

### 单 epoch 训练冒烟测试

无需下载数据集，脚本会临时生成 8 条 WAV 音频，并完整执行音频解码、Log-Mel
提取、动态批处理、CNN 前向传播、反向传播和参数更新：

```bash
python scripts/smoke_train_epoch.py
```

使用 CUDA：

```bash
python scripts/smoke_train_epoch.py --device cuda
```

成功时输出 `ONE_EPOCH_SMOKE_TEST=PASS`。临时数据会在运行结束后自动清理。

### 标准训练工作流

训练以版本化 `ExperimentConfig` 为配置入口：

```bash
ser dataset scan --importer folder --source "path/to/audio"
ser dataset import --importer folder --source "path/to/audio" --destination "data/standard"
ser dataset validate "data/standard/dataset.yaml" --check-files
ser dataset stats "data/standard/dataset.yaml" --probe-audio
ser train "configs/cnn_logmel.yaml" --split train --batch-size 16
```

仓库提供了可直接修改的 CNN、GRU 和 Transformer 配置模板，参见
[`configs/README.md`](configs/README.md)。Python API 示例位于
[`examples/`](examples/README.md)。

训练 checkpoint 只用于从可信的本地训练状态继续执行。需要发布或推理时，应先
导出为安全 artifact：

```bash
ser artifact export \
  --config "configs/cnn_logmel.yaml" \
  --checkpoint "runs/cnn-logmel/checkpoints/best.pt" \
  --destination "artifacts/model"
ser artifact verify "artifacts/model" --json
```

Windows PowerShell 可将以上多行命令写在一行，或使用反引号代替 `\` 续行。

### 离线与批量推理

```bash
ser predict "artifacts/model" "path/to/audio" \
  --output "runs/predictions.jsonl"
ser evaluate "artifacts/model" \
  --manifest "data/standard/dataset.yaml" \
  --split test \
  --output "runs/evaluation"
```

Python API：

```python
from ser_lib.artifacts import load_model_artifact
from ser_lib.inference import EmotionPredictor

loaded = load_model_artifact("artifacts/model")
predictor = EmotionPredictor(
    loaded.model,
    loaded.audio_loader,
    loaded.pipeline,
    loaded.collator,
    labels=loaded.manifest.labels,
)
result = predictor.predict_file("path/to/test.wav")
print(result.emotion, result.confidence, result.probabilities)
```

### 流式推理核心

流式模块接收 PCM tensor，不负责访问麦克风设备或提供界面：

```python
from ser_lib.inference import StreamingConfig, StreamingEmotionRecognizer

session = StreamingEmotionRecognizer(
    predictor,
    StreamingConfig(input_sample_rate=16000, window_ms=2000, hop_ms=500),
)
for window_result in session.push_pcm(pcm_chunk):
    if window_result.prediction is not None:
        print(window_result.start_ms, window_result.prediction.emotion)
session.close()
```

## 数据与模型如何解耦

项目不会让 Dataset 判断当前使用 CNN、RNN 或 Transformer。数据首先由
Representation 转换为带有明确 `TensorSpec` 的输入，再由模型通过 `ModelSpec`
声明需求，启动训练或推理前统一检查兼容性：

```text
外部数据
  → Importer / Manifest
  → AudioLoader
  → Representation + Transform
  → SERDataset
  → SERCollator
  → SERBatch
  → ModelSpec 兼容性检查
  → Train / Evaluate / Inference
```

因此，增加一种特征表示通常不需要新建 Dataset；增加模型也不应把模型判断写回
数据加载代码。

## 待开发功能

以下内容尚不能作为已发布能力使用：

- 将 0–7 章 Notebook 从课程提纲补齐为可执行、可由 CI 验证的教程。
- 增加更完整的损失函数插件协议，以及面向连续情感维度的回归损失。
- 建立 CPU/GPU 冷启动、吞吐、峰值内存和长时间流式运行的正式性能基线。
- 完成 IEMOCAP、MELD 等复杂会话型数据集 importer，并进行超大 manifest 压测。
- 完善第三方模型适配协议、多任务 valence/arousal 输出和更多受控模型族。
- 提供 artifact schema 迁移工具，以及按需求评估 ONNX 等独立导出格式。
- 补充发布自动化、macOS 真实运行验证和更严格的分模块覆盖率门禁。

桌面端、Web UI、麦克风设备管理、账户系统和产品服务端不在本仓库计划内；这些
能力应在独立应用仓库中通过 Python API、CLI 或专用适配层集成。

## 项目结构

```text
Speech-Emotion-Recognition/
├── ser_lib/
│   ├── core/             # 配置、异常、事件和日志
│   ├── data/             # 统一数据流水线与数据集导入
│   ├── models/           # 模型协议、注册表和内置模型
│   ├── engine/           # 训练、评估、优化器和 checkpoint
│   ├── artifacts/        # 安全模型产物导出与加载
│   ├── inference/        # 单文件、批量和流式推理
│   └── cli/              # ser 命令行入口
├── scripts/              # 覆盖率和训练冒烟脚本
├── configs/              # 可校验的训练配置模板
├── examples/             # 可执行 Python API 示例
├── benchmarks/           # 数据流水线性能基准
├── tests/                # 自动化测试
├── tutorials/            # 课程 Notebook（当前为提纲）
├── docs/                 # 设计、使用和 API 文档
├── data/                 # 本地数据说明与待迁移兼容脚本
└── pyproject.toml        # 包、依赖和工具配置
```

## 文档

- [安装与环境](docs/INSTALLATION.md)
- [标准数据格式](docs/DATA_FORMAT.md)
- [训练与 CLI](docs/TRAINING_AND_CLI.md)
- [模型扩展](docs/MODEL_DEVELOPMENT.md)
- [Artifact 与安全](docs/ARTIFACTS_AND_SECURITY.md)
- [公共 API](docs/API_REFERENCE.md)
- [仓库整体实施计划](docs/REPOSITORY_IMPLEMENTATION_PLAN.md)
- [教程状态](docs/TUTORIAL_STATUS.md)
- [变更记录](CHANGELOG.md)

## 兼容性与安全

- 当前支持 Python 3.10–3.12，主要目标平台为 Windows、Linux 和 macOS。
- 默认不加载不可信 pickle；分发模型应使用 artifact v2 和 `safetensors`。
- `hf_audio_classifier` 默认只加载本地文件并禁止远程自定义代码。
- 数据集和预训练模型受各自许可证约束，本仓库不会自动打包或重新分发它们。
- 发现安全问题时请按照 [安全策略](SECURITY.md) 私下报告。

## 贡献

欢迎提交数据集适配器、模型、测试、教程和文档改进。提交前请阅读
[贡献指南](CONTRIBUTING.md)，并确保没有提交真实数据集、大型权重、缓存、密钥或
机器相关绝对路径。

## 许可证

项目代码使用 [MIT License](LICENSE)。第三方依赖与数据集许可说明见
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
