# 语音情感识别 (Speech Emotion Recognition) 学习资源库

本项目旨在打造一个**从入门到进阶的语音情感识别 (SER) 学习资源与通用工具包**。无论是希望学习音频处理和深度学习的初学者，还是需要一个简单方便调用的情感识别 API 的开发者，都可以在本项目中找到所需的资源和代码。

本项目的核心理念分为两部分：
1. **通用处理库 (Core Library)**：封装标准化的音频提取流程、多种异构数据集加载器（MFCC, Mel-spectrogram等）及各类型深度学习模型接口。
2. **渐进式教程 (Tutorials)**：一套手把手的 Jupyter Notebook 教程，从零开始演示传统机器学习、CNN、RNN 直至最前沿的大模型在语音情感分类上的应用。

---

## 一、 数据集介绍与支持 (Supported Datasets)

语音情感识别依赖于高质量的数据集。本项目深度内置了对多种经典与前沿中英文语料库的支持与预处理流程。以下为本库处理的主要数据集：

### 1. ESD (Emotional Speech Dataset)
[GitHub 链接](https://github.com/HLTSingapore/Emotional-Speech-Data)
该数据集包含 10 位以普通话为母语的人和 10 位以英语为母语的人所说的平行语句。
* **语言**：中英文 (Mandarin & English)
* **说话人数量**：20 人 (10 位普通话母语者，10 位英语母语者)
* **情感类别 (5 种)**：Neutral (中性)、Happy (快乐)、Angry (愤怒)、Sad (悲伤)、Surprise (惊讶)
* **句子总数**：350 个平行语句 × 5种情绪 × 20人 = 35,000 句

### 2. IEMOCAP 
[GitHub 链接参考](https://github.com/tuncayka/speech_emotion)
交互式情感二元运动捕捉数据库，是最经典的英文多模态情感数据集。
* **语言**：英文 (English)
* **说话人数量**：10 人 (5男5女，分 5 个 Session 录制)
* **情感类别**：分类标签 (愤怒、快乐、悲伤、中性等) 及连续维度标签 (VAD)
* **句子总数**：10,039 句 (话语级别)
* **特点**：交互式对话录音，包含视频和动作捕捉数据

### 3. CREMA-D 
[GitHub 链接](https://github.com/CheyneyComputerScience/CREMA-D)
大规模的众包情感音视频数据集，覆盖广谱的年龄段和多种族群。
* **语言**：英文 (English)
* **说话人数量**：91 人 (48位男性和43位女性，年龄 20-74 岁，包含非裔、亚裔、白人、西班牙裔等)
* **情感类别 (6 种)**：愤怒、厌恶、恐惧、快乐、中性、悲伤
* **情绪强度 (4 种)**：低、中、高、未指明
* **句子总数**：7,442 个原始视频/音频片段 (演员们朗读了 12 个特定句子)

### 4. RAVDESS
[GitHub 链接参考](https://github.com/tuncayka/speech_emotion)
The Ryerson Audio-Visual Database of Emotional Speech and Song，高度规范的北美英语多模态情绪数据集。
* **语言**：英文 (English)
* **说话人数量**：24 人 (12 名男性，12 名女性，均为专业演员)
* **情感类别 (8 种)**：中性、平静、高兴、悲伤、愤怒、恐惧、厌恶、惊讶
* **情绪强度**：正常、强烈 (注: 中性情绪没有强烈强度)
* **数据集规模**：7,356 个文件 (含音视频)，由于单纯语音识别一般使用 Audio-only，纯语音文本部分包含 1,440 个文件
* **命名规则**：具有确切的 7 部分数字文件名标识体系 (如 `03-01-05-01-01-01-01.wav`)

### 5. CSEMOTIONS
[GitHub 链接](https://github.com/AIDC-AI/Marco-Voice/tree/main/Dataset)
CSEMOTIONS 是一个专为表现力语音合成、情感识别及声音克隆研究设计的高质量普通话 (Mandarin) 情感语音数据集。

| 属性 | 详细信息 |
| :--- | :--- |
| **语言** | 普通话 (Mandarin Chinese) |
| **总时长** | ~10.24 小时 |
| **说话人数量** | 6 人 (3 男，3 女，专业配音演员) |
| **情感类别 (7 种)**| 中性、快乐、愤怒、悲伤、惊讶、俏皮、恐惧 |
| **音频格式** | WAV, 单声道 (Mono), 48kHz / 24-bit PCM, 录音室级质量 |
| **句子总数** | 4,160 句 |

**CSEMOTIONS 情感分布如下表：**

| 情感标签 (Label) | 时长 (Duration) | 句子数量 (Sentences) |
| :--- | :--- | :--- |
| **Sad (悲伤)** | 1.73h | 546 |
| **Angry (愤怒)** | 1.43h | 769 |
| **Happy (快乐)** | 1.51h | 603 |
| **Surprise (惊讶)** | 1.25h | 508 |
| **Fearful (恐惧)** | 1.92h | 623 |
| **Playfulness (俏皮)** | 1.23h | 621 |
| **Neutral (中性)** | 1.14h | 490 |
| **总计 (Total)** | **10.24h** | **4,160** |

该数据集中的每句语音均配有中文文本转录、情感标签及说话人信息，并包含了中英文双语的评估提示词。这不仅适用于语音情感识别 (SER)，还适合用于跨语言合成实验。

---

## 二、 进阶路线与推荐项目架构 (Roadmap & Architecture)

为了方便初学者理解，本项目代码库设计为**模块化结构**，支持由浅入深的四级进阶学习路线：

1. **第一支柱：音频特征工厂 (Feature Engineering)**
   - 提取基础声学特征并在 Jupyter 中进行波形与二维谱图 (Spectrogram) 的可视化分析。
   - 统一异构数据集格式返回 PyTorch DataLoader 对象。
2. **Level 1：传统基准与特征理解 (Traditional Baseline)**
   - 基于提取的全局统计特征 (如 MFCCs)，使用 Scikit-Learn 训练随机森林 (RF) 或支持向量机 (SVM)。
3. **Level 2：二维视觉转化 (CNN Modeling)**
   - 将语音 1D 波形转为 2D 梅尔语谱图 (Mel-Spectrogram)，使用轻量级的 ResNet 结构进行图象分类维度的情感判别。
4. **Level 3：时序序列分析 (RNN/LSTM/GRU)**
   - 按帧提取时序特征流，训练 LSTM 获取语句前后的情感时序关联与表征。
5. **Level 4：预训练基座时代 (Pre-trained Models)**
   - 介绍与接入 HuggingFace `transformers` 库，微调诸如 Wav2Vec 2.0 / HuBERT 或 AST 等前沿 SOTA 音频模型对情感特征进行最终降维处理。

## 三、 快速使用 (Quick Start) *(规划中)*
最终发布的库将支持极致简化的接口调用：
```python
from ser_lib.inference import EmotionRecognizer

# 加载最好的预训练情感模型
recognizer = EmotionRecognizer(model_type='cnn_mel', language='zh') 

# 单行预测
result = recognizer.predict_audio("test_sample.wav")
print(f"预测情感: {result['emotion']}, 信心指数: {result['confidence']:.2f}")
```
