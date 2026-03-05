# 🎙️ Speech Emotion Recognition (SER)

一个将**工程实用性**与**零基础教学**完美结合的语音情感识别（SER）开源库。



## 🚀 快速开始 (Quick Start)

### 安装配置

克隆仓库后，推荐在虚拟环境中使用可编辑模式安装：

```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
pip install -e .
```

### 极简推理 API

只需 3 行代码即可完成一次离线音频的情感预测：

```python
from ser_lib.inference.offline import EmotionRecognizer

# 初始化识别器，默认加载基线模型
recognizer = EmotionRecognizer(model_type='cnn_mel', language='zh') 

# 传入音频文件路径进行预测
result = recognizer.predict_file("path/to/test_audio.wav")
print(f"检测到的情感是: {result['emotion']} (置信度: {result['confidence']:.2f})")

```

---

## 一、 数据集介绍与支持 (Supported Datasets)

详细的数据集介绍与支持列表，请参考 [数据目录说明](data/readme.md)。

---

## 📚 学习教程 (Tutorials)

如果你是语音处理或深度学习的初学者，请按顺序查阅 `tutorials/` 目录下的 Notebook，它们将带你从零实现 SER 库的核心功能：

* **【课程 0】语音数据与预处理（入门根基） (`00_audio_data_preprocess.ipynb`)** - 核心：音频格式统一、VAD 降噪、数据集分层划分。输出：`AudioPreprocessor` 批量预处理类。
* **【课程 1】传统 ML 基线：SER 入门基准 (`01_ml_baseline_mfcc_svm.ipynb`)** - 核心：MFCC 手工特征、SVM / 随机森林训练、基础评估。输出：基线模型推理脚本。
* **【课程 2】深度学习前置：PyTorch 工程基础 (`02_pytorch_data_pipeline.ipynb`)** - 核心：Dataset/DataLoader 封装、语音数据增强、通用训练模板。输出：`SERDataLoader` 与 `BaseTrainer` 通用类。
* **【课程 3】CNN：语音视觉化建模 (`03_cnn_ser_spectrogram.ipynb`)** - 核心：梅尔语谱图、2D-CNN 情感分类、模型推理。输出：`CNNSERInferencer` 推理类。
* **【课程 4】Transformer：时序情感建模 (`04_transformer_ser_temporal.ipynb`)** - 核心：注意力机制适配语音、轻量级 Transformer 训练。输出：Transformer SER 训练 / 推理脚本。
* **【课程 5】SER 主流混合模型（工业级落地方案） (`05_ser_mainstream_hybrid_models.ipynb`)** - 核心：CNN-Transformer 混合模型、ECAPA-TDNN、TCN、模型选型对比。输出：主流模型统一训练 / 推理模板、性能对比表。
* **【课程 6】预训练模型与微调（SOTA 方案） (`06_ser_pretrained_model_finetune.ipynb`)** - 核心：预训练模型选型及微调全流程验证。输出：预训练模型微调一键脚本。
* **【课程 7】工程实践：解决实际场景数据偏差 (`07_ser_engineering_practice_bias.ipynb`)** - 核心：训练 / 工程数据分布偏移、数据 / 模型层解决方案、A/B 测试。输出：工程偏差分析工具。
* **【课程 8】可视化交互（工具化落地） (`08_ser_visualization_interaction.ipynb`)** - 核心：结果可视化图表、交互式测试界面、教学可视化工具及轻量化部署。输出：可视化分析工具、交互式测试界面。