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

* **Level 1: 基准与特征工程 (`01_baseline_ml.ipynb`)** - 使用 `librosa` 提取 MFCC 特征，并训练 SVM 模型。
* **Level 1.5: 深度学习之桥 (`02_pytorch_bridge.ipynb`)** - 手写 PyTorch Dataset 和 DataLoader，跨越认知悬崖。
* **Level 2: 图像化语音 (`03_cnn_spectrogram.ipynb`)** - 将音频转化为梅尔语谱图并使用 CNN 进行图像分类。
* **Level 3: 捕捉时间信息 (`04_lstm_sequence.ipynb`)** - 引入 LSTM 网络处理音频的时间序列特性。
* **Level 4: 预训练大模型 (`05_transformer_finetune.ipynb`)** - 使用 HuggingFace 微调 Wav2Vec2 / HuBERT 等 SOTA 模型。
* **Level 5: 评估与模型去偏 (`06_evaluation_bias.ipynb`)** - 掌握留一法交叉验证 (LOSO) 与混淆矩阵解读。
* **Level 6: 模型部署 (`07_gradio_deployment.ipynb`)** - 使用 Gradio 构建可互动的网页麦克风测试接口。