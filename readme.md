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

语音情感识别依赖于高质量的数据集。本项目深度内置了对多种经典与前沿中英文语料库的支持与预处理流程。以下为本库处理的主要数据集：

### 1. ESD (Emotional Speech Dataset)

[GitHub 链接](https://github.com/HLTSingapore/Emotional-Speech-Data)
该数据集包含 10 位以普通话为母语的人和 10 位以英语为母语的人所说的平行语句。

* **语言**：中英文 (Mandarin & English)
* **说话人数量**：20 人 (10 位普通话母语者，10 位英语母语者)
* **情感类别 (5 种)**：Neutral (中性)、Happy (快乐)、Angry (愤怒)、Sad (悲伤)、Surprise (惊讶)
* **句子总数**：350 个平行语句 × 5种情绪 × 20人 = 35,000 句

### 2. IEMOCAP

[GitHub 链接](https://github.com/tuncayka/speech_emotion)
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

[GitHub 链接](https://github.com/tuncayka/speech_emotion)
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
| --- | --- |
| **语言** | 普通话 (Mandarin Chinese) |
| **总时长** | ~10.24 小时 |
| **说话人数量** | 6 人 (3 男，3 女，专业配音演员) |
| **情感类别 (7 种)** | 中性、快乐、愤怒、悲伤、惊讶、俏皮、恐惧 |
| **音频格式** | WAV, 单声道 (Mono), 48kHz / 24-bit PCM, 录音室级质量 |
| **句子总数** | 4,160 句 |

**CSEMOTIONS 情感分布如下表：**

| 情感标签 (Label) | 时长 (Duration) | 句子数量 (Sentences) |
| --- | --- | --- |
| **Sad (悲伤)** | 1.73h | 546 |
| **Angry (愤怒)** | 1.43h | 769 |
| **Happy (快乐)** | 1.51h | 603 |
| **Surprise (惊讶)** | 1.25h | 508 |
| **Fearful (恐惧)** | 1.92h | 623 |
| **Playfulness (俏皮)** | 1.23h | 621 |
| **Neutral (中性)** | 1.14h | 490 |
| **总计 (Total)** | **10.24h** | **4,160** |

该数据集中的每句语音均配有中文文本转录、情感标签及说话人信息，并包含了中英文双语的评估提示词。这不仅适用于语音情感识别 (SER)，还适合用于跨语言合成实验。

### 6. BAAI-Emotiontalk

[GitHub 链接](https://github.com/NKU-HLT/EmotionTalk) | [HuggingFace 链接](https://huggingface.co/datasets/BAAI/Emotiontalk)
EmotionTalk 是一个具有丰富标注的交互式中文多模态情感数据集。该数据集提供了 19 位演员参与二元对话场景的多模态信息，融合了声学、视觉和文本模态。

* **语言**：中文 (Chinese)
* **总时长**：23.6 小时 (19,250 句话语的自发性对话录音)
* **说话人数量**：19 人
* **音频格式**：WAV 文件 (44.1kHz 采样率)
* **情感类别 (7 种)**：Happy (快乐)、Angry (愤怒)、Sad (悲伤)、Disgust (厌恶)、Fear (恐惧)、Surprise (惊讶)、Neutral (中性)
* **细粒度标注**：包含 5 维情绪区间标签（消极、弱消极、中性、弱积极、积极）与 4 维语音描述（说话人、说话风格、情绪及整体表现）

**数据集分布如下表：**

| 数据划分 | Angry (愤怒) | Disgusted (厌恶) | Fearful (恐惧) | Happy (快乐) | Neutral (中性) | Sad (悲伤) | Surprised (惊讶) | 总计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Train** | 2950 | 1142 | 672 | 2986 | 5377 | 919 | 1367 | 15413 |
| **Val(G01/G12)** | 409 | 95 | 125 | 360 | 675 | 111 | 133 | 1908 |
| **Test(G03/G15)** | 339 | 134 | 125 | 246 | 801 | 123 | 161 | 1929 |
| **Total** | 3698 | 1371 | 922 | 3592 | 6853 | 1153 | 1661 | 19250 |


### 7. CASIA-Emotional Speech Dataset

CASIA 数据集采用三阶段采集流程，是一个经过严格质量控制的中文情感语音数据集。

* **数据采集**：通过众包及声学实验室（16kHz，16bit）录制标准化语音，包含指令性任务和自由对话。覆盖 1,200 名（18-65岁）来自不同方言区的说话人。
* **数据标注**：使用 Praat 提取特征，由双盲标注确保情感标签一致性（Kappa 系数 > 0.85）。

**核心数据特征如下表：**

| 特征维度 | 具体指标 |
| --- | --- |
| **样本规模** | 20,480 条语音 (训练集 16,384条，验证集 2,048条，测试集 2,048条) |
| **时长分布** | 平均每条 3.2 秒，最短 1.5 秒，最长 8 秒 |
| **情感分布** | 中性 (32%)、高兴 (18%)、悲伤 (15%)、愤怒 (12%)、惊讶 (10%)、恐惧 (8%)、复合情绪 (5%) |
| **声学参数** | 基频范围 80-450Hz，能量动态范围 -30dB 至 0dB，语速 60-300字/分钟 |

### 8. MELD Dataset

[官方地址](http://affective-meld.github.io)
MELD (Multimodal EmotionLines Dataset) 是一款面向**对话情感识别 (ERC)** 任务的多模态、多方对话情感数据集，由新加坡科技设计大学等机构构建。该数据集填补了大尺度多模态多方对话情感数据库的空白。

* **数据来源**：提取自经典美剧《Friends（老友记）》
* **数据规模**：包含 1,433 个对话，总计约 13,000 条话语。
* **模态支持**：文本、音频、视觉三模态。
* **情感类别 (7 种)**：愤怒 (Anger)、厌恶 (Disgust)、恐惧 (Fear)、喜悦 (Joy)、中性 (Neutral)、悲伤 (Sadness)、惊讶 (Surprise)。
* **情感倾向 (3 种)**：积极、消极、中性。
* **数据特征**：
  * **对话特征**：每个对话平均包含 9.5 条话语，2.7 名说话者，最多支持 9 人多方对话。
  * **情感动态**：说话者的情感转移现象频繁，是多方对话情感识别的核心挑战。
* **音频格式**：16 位 PCM WAV 文件。

**数据集划分与分布：**

| 数据划分 | 对话数量 | 话语数量 |
| --- | --- | --- |
| **Train** | 1039 | 9989 |
| **Val** | 114 | 1109 |
| **Test** | 280 | 2610 |
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