## 一、 数据集介绍与支持 (Supported Datasets)

本目录中的数据处理脚本统一输出新数据流水线格式，不再生成或使用按输入表示
拆分的 Dataset 类。以 CASIA 为例：

```bash
python data/casia_process.py
```

默认输出到 `configs/datasets/casia/`：

```text
dataset.yaml          # 数据集根目录、划分和标签表
train.jsonl
val.jsonl
test.jsonl
data_waveform.yaml    # 原始波形 Representation
data_log_mel.yaml     # Log-Mel Representation
data_acoustic.yaml    # 声学组合 Representation
data_report.md
```

加载时统一使用 `ser_lib.data`：

```python
from ser_lib.data import DatasetManifest, SERDataset, load_data_config
from ser_lib.data import build_components, build_collator

config = load_data_config("configs/datasets/casia/data_log_mel.yaml")
manifest = DatasetManifest.load(config.manifest)
loader, pipeline = build_components(config, train=True)
dataset = SERDataset(manifest.resolved_records("train"), loader, pipeline)
collator = build_collator(pipeline.output_specs, config.batching)
```

语音情感识别依赖于高质量的数据集。本项目深度内置了对多种经典与前沿中英文语料库的支持与预处理流程。以下为本库处理的主要数据集：

### 1. ESD (Emotional Speech Dataset)

[GitHub 链接](https://github.com/HLTSingapore/Emotional-Speech-Data)
该数据集包含 10 位以普通话为母语的人和 10 位以英语为母语的人所说的平行语句。

* **语言**：中英文 (Mandarin & English)
* **说话人数量**：20 人 (10 位普通话母语者，10 位英语母语者)
* **情感类别 (5 种)**：Neutral (中性)、Happy (快乐)、Angry (愤怒)、Sad (悲伤)、Surprise (惊讶)
* **句子总数**：350 个平行语句 × 5种情绪 × 20人 = 35,000 句

本地数据可直接转换为标准 manifest。默认分别在中文和英文说话人中按 60%/20%/20%
划分 train/val/test，确保三组说话人互斥，同时保持语言构成一致：

```bash
ser dataset scan --importer esd --source "data/Emotion Speech Dataset" --json
ser dataset import --importer esd --source "data/Emotion Speech Dataset" \
  --destination data/esd-standard
ser dataset validate data/esd-standard/dataset.yaml --check-files
ser train configs/esd_cnn_logmel.yaml --batch-size 32
```

Importer 会读取每位说话人的 `<speaker>.txt` 转写文件，保留文本、语言、情感和说话人
字段。可通过 `languages: [zh]` 或 `languages: [en]` 只导入一种语言，也可使用
`speaker_splits` 明确指定实验划分。

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

本地 `AudioWAV` 可通过专用 importer 转换。默认结合 `VideoDemographics.csv`
按性别分层、按演员互斥划分数据，并保留年龄、性别、族裔、固定语句和情感强度：

```bash
ser dataset scan --importer crema_d --source data/CREMA-D --json
ser dataset import --importer crema_d --source data/CREMA-D \
  --destination data/crema-d-standard
ser dataset validate data/crema-d-standard/dataset.yaml --check-files
ser train configs/crema_d_cnn_logmel.yaml --batch-size 32
```

默认分类标签来自文件名中演员被要求表达的情感。它与 CREMA-D 众包语音感知投票
不是同一种监督信号，报告实验时应明确标注，不能把两者结果直接混用。

### 4. RAVDESS

[官方发布页](https://zenodo.org/records/1188976)
The Ryerson Audio-Visual Database of Emotional Speech and Song，高度规范的北美英语多模态情绪数据集。

* **语言**：英文 (English)
* **说话人数量**：24 人 (12 名男性，12 名女性，均为专业演员)
* **情感类别 (8 种)**：中性、平静、高兴、悲伤、愤怒、恐惧、厌恶、惊讶
* **情绪强度**：正常、强烈 (注: 中性情绪没有强烈强度)
* **数据集规模**：7,356 个文件 (含音视频)，由于单纯语音识别一般使用 Audio-only，纯语音文本部分包含 1,440 个文件
* **命名规则**：具有确切的 7 部分数字文件名标识体系 (如 `03-01-05-01-01-01-01.wav`)
* **许可**：CC BY-NC-SA 4.0；商业使用需另行确认官方许可

本库不会下载 RAVDESS。取得数据并确认许可后，可通过统一 CLI 导入：

```bash
ser dataset scan --importer ravdess --source "path/to/RAVDESS" --json
ser dataset import --importer ravdess --source "path/to/RAVDESS" --destination "data/ravdess-standard"
ser dataset stats "data/ravdess-standard/dataset.yaml" --probe-audio --json
```

本地 CASIA 可以使用说话人独立划分脚本整理，原始音频不会被移动或修改：

```bash
python scripts/prepare_casia.py --source data/CASIA --destination data/casia-standard
ser dataset validate data/casia-standard/dataset.yaml --check-files
ser train configs/casia_cnn_logmel.yaml --batch-size 32
```

脚本按说话人而不是按音频随机划分 train/val/test，避免同一说话人同时出现在训练
和评估集合中。四说话人版本使用 2/1/1 划分，适合验证完整链路，不足以得出稳定的
通用模型性能结论。

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

本地数据可通过专用 importer 转为说话人独立的标准 manifest：

```bash
ser dataset scan --importer csemotions --source data/CSEMOTIONS-data --json
ser dataset import --importer csemotions --source data/CSEMOTIONS-data \
  --destination data/csemotions-standard
ser dataset validate data/csemotions-standard/dataset.yaml --check-files
```

Importer 读取官方 `csemotions_metadata.csv`，保留文本、时长、说话人、语言和性别
元数据。默认在每个性别组内确定性地按约 60%/20%/20% 分配说话人，避免说话人
跨 split；也可通过 `speaker_splits` 显式指定。

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

本地发布文件经过 importer 实测后的最终标签分布为：Neutral 9,378、Angry 3,820、
Happy 2,105、Surprised 1,363、Sad 1,110、Disgusted 818、Fearful 656。不同发布版本
或二次整理版本可能采用不同标签统计，实验报告应以生成 manifest 的统计结果为准。

本地 JSON/WAV 可通过专用 importer 转换。默认按 `speaker_id` 划分，避免同一说话人
跨 train/val/test；也可选择数据集发布方的对话划分：

```bash
ser dataset scan --importer emotiontalk --source data/BAAI_Emotiontalk --json
ser dataset import --importer emotiontalk --source data/BAAI_Emotiontalk \
  --destination data/emotiontalk-standard
ser dataset validate data/emotiontalk-standard/dataset.yaml --check-files
ser train configs/emotiontalk_cnn_logmel.yaml --batch-size 32
```

Importer 保留文本、对话与轮次标识、时间信息、五位标注者的情感及置信度，以及
`sourceAttr` 中的语音描述。使用 `split_strategy: official_dialogue` 可以复现发布方
对话划分，但其中部分 `speaker_id` 会跨 split，因此严格说话人泛化实验应使用默认策略。
本地全量探测还显示音频同时存在 44.1 kHz/16 kHz 和双声道/单声道；训练配置中的
统一单声道及 16 kHz 重采样是必要步骤，而不是可省略的优化。


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
