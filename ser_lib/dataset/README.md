# SER Dataset Module

语音情感识别系统的数据集加载模块，采用配置驱动设计，支持多种特征提取策略与数据增强流水线。

---

## 1. 模块概述 (Overview)

### 1.1 设计目标

本模块为 SER 任务提供灵活的数据管道，核心特性：

- **配置驱动**：通过 YAML 配置文件控制数据集行为，无需修改代码
- **Schema 校验**：所有配置在 Dataset 初始化前通过 `config_schema.py` 做运行时校验
- **懒加载**：支持片段级音频加载，避免长音频的内存开销
- **多级增强**：波形级 → 谱图级 → 批次级的三层增强架构
- **多策略对齐**：支持固定长度填充、动态掩码、滑动窗口三种批处理策略

当前第一轮约束：`FeatureDataset` 仅支持 `dynamic_mask` 与 `truncate_pad`，暂不支持 `sliding_window`。

### 1.2 数据集类型

| 数据集类 | 输出格式 | 适用模型 | 配置模板 |
|---------|---------|---------|---------|
| `WaveformDataset` | 原始波形 `[T]` | Wav2Vec2, HuBERT, RawNet | `waveform_dataset.yaml` |
| `SpectrogramDataset` | 单一谱图 `[Freq, T]` | 2D CNN (ResNet, EfficientNet) | `spectrogram_dataset.yaml` |
| `FeatureDataset` | 多特征字典 `{name: Tensor}` | 自定义特征融合架构 | `feature_dataset.yaml` |

### 1.3 支持的数据集

当前已配置 **CASIA** 中文情感语音库：

| 属性 | 值 |
|-----|-----|
| 总样本数 | 1200 |
| 总时长 | 0.64 小时 (2286.78 秒) |
| 说话人数 | 4 人 |
| 情感类别 | 6 类 |
| 数据划分 | Train: 900 / Val: 150 / Test: 150 |

**情感标签映射：**

| ID | 英文 | 中文 |
|----|------|------|
| 0 | Neutral | 平静 |
| 1 | Happy | 高兴 |
| 2 | Angry | 愤怒 |
| 3 | Sad | 悲伤 |
| 4 | Surprise | 惊吓 |
| 5 | Fear | 恐惧 |

---

## 2. 预处理与特征工程 (Preprocessing & Feature Extraction)

### 2.1 音频加载流水线

```
Audio File (.wav)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  BaseConfigDataset._load_waveform()                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. 片段级懒加载 (可选)                           │   │
│  │    - frame_offset = start_time_ms / 1000 * sr   │   │
│  │    - num_frames = (end - start) / 1000 * sr     │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ 2. 单声道转换                                    │   │
│  │    waveform = mean(channels, dim=0, keepdim=True)│   │
│  ├─────────────────────────────────────────────────┤   │
│  │ 3. 重采样到目标采样率                            │   │
│  │    T.Resample(orig_sr → target_sr)              │   │
│  └─────────────────────────────────────────────────┘   │
│  Output: Tensor[1, T]                                  │
└─────────────────────────────────────────────────────────┘
```

**核心参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_sr` | 16000 Hz | 全局统一目标采样率 |
| `start_time_ms` | 0 | 片段起始时间（元数据字段） |
| `end_time_ms` | None | 片段结束时间（元数据字段） |

### 2.2 特征提取参数

**谱图特征核心参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_fft` | 1024 | FFT 窗口大小 |
| `win_length` | 400 | 窗函数长度 |
| `hop_length` | 256 | 帧移步长 |
| `n_mels` | 80 | Mel 滤波器组数量 |
| `f_min` | 0.0 Hz | 最低频率 |
| `f_max` | 8000.0 Hz | 最高频率 |
| `power` | 2.0 | 能量谱指数 |
| `top_db` | 80.0 | Log-Mel 动态范围裁剪 |

**时域韵律特征参数：**

| 特征 | 关键参数 | 说明 |
|------|----------|------|
| F0 | `freq_low=50, freq_high=800` | 基频检测范围 |
| RMS | `win_length=400, hop_length=256` | 短时能量 |
| ZCR | `win_length=400, hop_length=256` | 过零率 |

### 2.3 支持的特征类型

**谱图类特征 (`features/spectrogram.py`)：**

| 类型 | 输出维度 | 说明 |
|------|----------|------|
| `Spectrogram` | `[Freq, T]` | 线性谱图 |
| `MelSpectrogram` | `[n_mels, T]` | Mel 谱图 |
| `LogMelSpectrogram` | `[n_mels, T]` | Log-Mel 谱图（经 AmplitudeToDB） |
| `MFCC` | `[n_mfcc, T]` | Mel 频率倒谱系数 |

**时域特征 (`features/time_domain.py`)：**

| 特征 | 输出维度 | 说明 |
|------|----------|------|
| `F0` | `[1, T]` | 基频（基于 NCCF 算法） |
| `RMS` | `[1, T_frames]` | 短时能量 |
| `ZCR` | `[1, T_frames]` | 过零率 |
| `JitterShimmerHNR` | `[3]` | 音质特征（语句级标量） |

**频域特征 (`features/freq_domain.py`)：**

| 特征 | 输出维度 | 说明 |
|------|----------|------|
| `SpectralCentroid` | `[B, Time]` | 谱质心（声音明亮度） |
| `SpectralRolloff` | `[B, Time]` | 谱滚降（85% 能量阈值频率） |
| `SpectralFlatness` | `[B, Time]` | 谱平坦度（纯音 vs 噪声） |
| `SpectralFlux` | `[B, Time]` | 谱通量（起音检测） |
| `Delta` | `[B, C*3, Time]` | 一阶/二阶差分拼接 |

---

## 3. 数据结构与维度 (Data Structures & Shapes)

### 3.1 单样本输出格式

**`__getitem__` 返回值：**

```python
Tuple[Dict[str, torch.Tensor], torch.Tensor, int]
#     ↑ features_dict      ↑ label      ↑ seq_length
```

| 数据集类 | features_dict 内容 | seq_length 含义 |
|---------|-------------------|-----------------|
| `WaveformDataset` | `{"raw_waveform": [T]}` | 采样点数 |
| `SpectrogramDataset` | `{"<spec_type>": [Freq, T]}` | 时间帧数 |
| `FeatureDataset` | `{"f0": [T], "rms": [T], ...}` | 首个特征的时间帧数 |

### 3.2 批处理输出格式

**`collate_fn` 返回值：**

```python
{
    "inputs": Dict[str, torch.Tensor],
    "labels": torch.Tensor,
    "lengths": torch.Tensor,
    "mask": Optional[torch.Tensor],
    "meta": Dict[str, Any],
}
```

- `inputs`：模型输入张量集合
- `labels`：当前 batch 对应标签
- `lengths`：真实有效长度（或窗口展开后的长度）
- `mask`：仅 `dynamic_mask` 返回布尔掩码，其余策略为 `None`
- `meta`：批处理元信息，至少包含 `dataset_type` 与 `collate_strategy`

**策略一：`truncate_pad`（固定长度）**

适用于 2D CNN 模型，输出固定尺寸张量：

```python
{
    'inputs': {
        'raw_waveform': Tensor[B, T_fixed]             # WaveformDataset
        '<feature_name>': Tensor[B, T_fixed, D]        # FeatureDataset
        '<spec_name>': Tensor[B, 1, Freq, T_fixed]     # SpectrogramDataset
    },
    'labels': Tensor[B],
    'lengths': Tensor[B],
    'mask': None,
    'meta': {...}
}
```

- 超过 `max_frames` 的截断，不足的右侧补零
- 默认 `max_frames=300`（约 3 秒语音）

**策略二：`dynamic_mask`（动态长度）**

适用于 Transformer 类模型，输出变长序列 + 注意力掩码：

```python
{
    'inputs': {
        'raw_waveform': Tensor[B, T_max],
        '<feature_name>': Tensor[B, T_max, D],
        '<spec_name>': Tensor[B, T_max, Freq]
    },
    'labels': Tensor[B],
    'lengths': Tensor[B],
    'mask': Tensor[B, T_max],  # True=真实数据, False=填充区域
    'meta': {...}
}
```

**策略三：`sliding_window`（滑动窗口）**

适用于长音频评估，按窗口切分后投票：

```python
{
    'inputs': {
        'raw_waveform': Tensor[B_expanded, window_size],
        '<spec_name>': Tensor[B_expanded, 1, Freq, window_size]
    },
    'labels': Tensor[B_expanded],
    'lengths': Tensor[B_expanded],
    'mask': None,
    'meta': {
        'window_counts': Tensor[B],     # 每个原始样本展开的窗口数
        'original_labels': Tensor[B]    # 原始标签（用于投票聚合）
    }
}
```

- 默认 `window_size=300, stride=150`（50% 重叠）

### 3.3 数据流完整维度变化

以 `SpectrogramDataset` + `truncate_pad` 为例：

```
原始音频文件 (.wav)
    │
    ▼ _load_waveform()
Tensor[1, T_samples]           # T_samples = duration * 16000
    │
    ▼ 时域增强 (可选)
Tensor[1, T_samples]
    │
    ▼ LogMelSpectrogram 提取
Tensor[1, n_mels, T_frames]    # T_frames = T_samples / hop_length
    │
    ▼ 频域增强 (可选)
Tensor[1, n_mels, T_frames]
    │
    ▼ squeeze(0) → _load_item() 返回
Tensor[n_mels, T_frames]
    │
    ▼ collate_fn (truncate_pad)
Tensor[B, 1, n_mels, max_frames]
```

---

## 4. 核心类与方法 (Core Classes & Methods)

### 4.1 类继承结构

```
torch.utils.data.Dataset
        │
        └── BaseConfigDataset (抽象基类)
                │
                ├── WaveformDataset
                │
                ├── SpectrogramDataset
                │
                └── FeatureDataset
```

### 4.2 BaseConfigDataset

**文件：** `base_dataset.py`

**职责：** 配置解析、数据列表加载、音频懒加载、重采样管理

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `split` | `str` | 数据集划分 (`'train'`/`'val'`/`'test'`) |
| `config` | `dict` | 解析后的 YAML 配置 |
| `target_sr` | `int` | 目标采样率 |
| `data_list` | `List[dict]` | 元数据列表 |
| `id2label` | `dict` | 标签 ID → 名称映射 |
| `resamplers` | `Dict[int, nn.Module]` | 采样率 → 重采样器缓存 |

**核心方法：**

| 方法 | 入参 | 出参 | 说明 |
|------|------|------|------|
| `__init__` | `config_path: str, split: str` | - | 初始化配置与数据列表 |
| `__len__` | - | `int` | 返回样本总数 |
| `__getitem__` | `idx: int` | `Tuple[Dict, Tensor, int]` | 获取单样本 |
| `_load_waveform` | `item: dict` | `Tensor[1, T]` | 懒加载音频波形 |
| `_load_item` | `waveform: Tensor, item: dict` | `Tuple[Dict, int]` | **抽象方法**，子类实现 |
| `get_labels` | - | `List[int]` | 返回全部标签（用于类别权重计算） |

**`_load_waveform` 实现细节：**

```python
def _load_waveform(self, item: dict) -> torch.Tensor:
    # 1. 路径解析（支持相对路径拼接 data_root_dir）
    audio_path = item['audio_path']

    # 2. 片段级加载（避免长音频内存开销）
    if start_time_ms > 0 or end_time_ms is not None:
        frame_offset = int((start_time_ms / 1000.0) * orig_sr)
        num_frames = int(((end_time_ms - start_time_ms) / 1000.0) * orig_sr)
        waveform, sr = torchaudio.load(audio_path, frame_offset, num_frames)

    # 3. 单声道转换
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. 重采样（带缓存）
    if sr != self.target_sr:
        if sr not in self.resamplers:
            self.resamplers[sr] = T.Resample(sr, self.target_sr)
        waveform = self.resamplers[sr](waveform)

    return waveform  # [1, T]
```

### 4.3 WaveformDataset

**文件：** `waveform_dataset.py`

**职责：** 加载原始波形，应用时域增强，不做特征提取

**`_load_item` 实现：**

| 入参 | 类型 | 说明 |
|------|------|------|
| `waveform` | `Tensor[1, T]` | 原始波形 |
| `item` | `dict` | 元数据（未使用） |

| 出参 | 类型 | 说明 |
|------|------|------|
| `features` | `Dict[str, Tensor]` | `{"raw_waveform": Tensor[T]}` |
| `seq_length` | `int` | 采样点数 |

```python
def _load_item(self, waveform, item):
    if self.wave_transform:
        waveform = self.wave_transform(waveform)
    if self.adv_wave_transform:
        waveform = self.adv_wave_transform(waveform)
    raw = waveform.squeeze(0)  # [T]
    return {"raw_waveform": raw}, raw.shape[-1]
```

### 4.4 SpectrogramDataset

**文件：** `spectrogram_dataset.py`

**职责：** 提取单一谱图特征，支持频域增强

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `spec_type_name` | `str` | 谱图类型名称 |
| `extractor` | `nn.Module` | 谱图提取器 |
| `wave_transform` | `nn.Module` | 时域增强流水线 |
| `spec_transform` | `nn.Module` | 频域增强流水线 |

**`_load_item` 实现：**

| 入参 | 类型 | 说明 |
|------|------|------|
| `waveform` | `Tensor[1, T]` | 原始波形 |
| `item` | `dict` | 元数据（未使用） |

| 出参 | 类型 | 说明 |
|------|------|------|
| `features` | `Dict[str, Tensor]` | `{spec_type_name.lower(): Tensor[Freq, T]}` |
| `seq_length` | `int` | 时间帧数 |

```python
def _load_item(self, waveform, item):
    # 1. 时域增强
    if self.wave_transform:
        waveform = self.wave_transform(waveform)
    if self.adv_wave_transform:
        waveform = self.adv_wave_transform(waveform)

    # 2. 谱图提取
    feat = self.extractor(waveform)

    # 3. 频域增强
    if self.spec_transform:
        feat = self.spec_transform(feat)

    feat = feat.squeeze(0)  # [Freq, T]
    return {self.spec_type_name.lower(): feat}, feat.shape[-1]
```

### 4.5 FeatureDataset

**文件：** `feature_dataset.py`

**职责：** 多特征并发提取，按字典返回，不强制拼接

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `extractors` | `nn.ModuleDict` | 特征名 → 提取器映射 |

**`_load_item` 实现：**

| 入参 | 类型 | 说明 |
|------|------|------|
| `waveform` | `Tensor[1, T]` | 原始波形 |
| `item` | `dict` | 元数据（未使用） |

| 出参 | 类型 | 说明 |
|------|------|------|
| `features` | `Dict[str, Tensor]` | 多特征字典 |
| `seq_length` | `int` | 首个特征的时间帧数 |

```python
def _load_item(self, waveform, item):
    # 1. 时域增强
    if self.wave_transform:
        waveform = self.wave_transform(waveform)
    if self.adv_wave_transform:
        waveform = self.adv_wave_transform(waveform)

    # 2. 逐特征提取
    features = {}
    for feat_name, extractor in self.extractors.items():
        feat = extractor(waveform).squeeze(0)
        features[feat_name] = feat

    seq_length = list(features.values())[0].shape[-1]
    return features, seq_length
```

### 4.6 build_collate_fn

**文件：** `collate.py`

**职责：** 根据配置生成批处理函数

**入参：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `dict` | 解析后的 YAML 配置 |

**出参：**

| 类型 | 说明 |
|------|------|
| `Callable` | 供 `DataLoader` 使用的 `collate_fn` |

**配置项：**

```yaml
audio_processing:
  strategy: "truncate_pad"  # truncate_pad | dynamic_mask | sliding_window
  max_frames: 300           # truncate_pad 专用
  window_size: 300          # sliding_window 专用
  stride: 150
```

---

## 5. 数据增强 (Data Augmentation)

### 5.1 增强层级架构

```
┌─────────────────────────────────────────────────────────────┐
│  Level 1: waveform_level (时域基础增强)                      │
│  ├─ AddGaussianNoise  (SNR 控制)                            │
│  ├─ PitchShift        (音调偏移)                            │
│  ├─ TimeStretch       (变速不变调)                          │
│  ├─ TimeShift         (时间平移)                            │
│  └─ VolumeScale       (音量缩放)                            │
├─────────────────────────────────────────────────────────────┤
│  Level 2: advanced_waveform_level (时域高级增强)             │
│  ├─ RIRSimulation     (房间冲激响应混响)                     │
│  └─ DynamicSNRMixing  (动态背景噪声混合)                     │
├─────────────────────────────────────────────────────────────┤
│  Level 3: spectrogram_level (频域增强)                       │
│  ├─ SpecMasking       (SpecAugment: 时频掩码)               │
│  ├─ FilterAugment     (随机频带滤波)                        │
│  └─ VTLP               (声道长度扰动)                        │
├─────────────────────────────────────────────────────────────┤
│  Level 4: batch_level (批次级增强)                           │
│  └─ Mixup             (样本混合 + 软标签)                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 时域增强类 (`augment/time_domain.py`)

| 类名 | 核心参数 | 说明 |
|------|----------|------|
| `AddGaussianNoise` | `snr=15.0, p=0.5` | 按 SNR 注入高斯白噪声 |
| `PitchShift` | `n_steps=4, p=0.5` | 音调偏移（半音阶），使用 Phase Vocoder |
| `TimeStretch` | `rate=1.2, p=0.5` | 变速不变调，需 STFT → Phase Vocoder → ISTFT |
| `TimeShift` | `shift_max_ratio=0.2, p=0.5` | 时间平移，空缺处补零 |
| `VolumeScale` | `gain_min=0.5, gain_max=1.5, p=0.5` | 随机音量缩放 |
| `RIRSimulation` | `rir_path, p=0.5` | 房间冲激响应混响（需外部 RIR 数据库） |
| `DynamicSNRMixing` | `noise_dataset_path, p=0.5` | 动态背景噪声混合（需外部噪声库） |

### 5.3 频域增强类 (`augment/freq_domain.py`)

| 类名 | 核心参数 | 说明 |
|------|----------|------|
| `SpecMasking` | `time_mask_param=30, freq_mask_param=15, p=0.5` | SpecAugment 时频掩码 |
| `FilterAugment` | `n_band=1, db_range=(-5, 5), p=0.5` | 随机频带增益（Log-Mel 域加法） |
| `VTLP` | `warp_factor_range=(0.9, 1.1), p=0.5` | 声道长度扰动（频率轴拉伸） |
| `SpecMix` | `p=0.5` | CutMix（需在 batch_level 实现） |

### 5.4 配置示例

```yaml
transforms:
  waveform_level:
    train:
      - type: "AddGaussianNoise"
        snr: 15.0
        p: 0.5
      - type: "TimeStretch"
        rate: 1.2
        p: 0.3
    val: []
    test: []

  spectrogram_level:
    train:
      - type: "SpecMasking"
        time_mask_param: 40
        freq_mask_param: 15
        p: 0.5
    val: []
    test: []
```

---

## 6. 配置文件规范 (Configuration Schema)

### 6.1 数据集配置结构

```yaml
# 元数据
dataset_name: "CASIA"
num_classes: 6
class_mapping:
  0: { en: "Neutral", zh: "平静" }
  1: { en: "Happy", zh: "高兴" }
  # ...

# 路径配置
paths:
  metadata_dir: "ser_lib/dataset/configs/casia"  # JSONL 文件目录
  data_root_dir: "data/CASIA"                    # 音频文件根目录

data_lists:
  train: "train.jsonl"
  val: "val.jsonl"
  test: "test.jsonl"

# 音频参数
audio:
  target_sr: 16000

# 批处理策略
audio_processing:
  strategy: "truncate_pad"
  max_frames: 300
  window_size: 300
  stride: 150

# 特征配置 (SpectrogramDataset 专用)
spectrogram:
  type: "LogMelSpectrogram"
  kwargs:
    n_fft: 1024
    hop_length: 256
    n_mels: 80
    f_min: 0.0
    f_max: 8000.0

# 或多特征配置 (FeatureDataset 专用)
features:
  selected_features:
    f0: { type: "F0" }
    rms: { type: "RMS" }
    zcr: { type: "ZCR" }

# 数据增强
transforms:
  waveform_level: { train: [], val: [], test: [] }
  spectrogram_level: { train: [], val: [], test: [] }
  batch_level: { train: [] }
```

### 6.2 JSONL 元数据格式

每行一个 JSON 对象：

```json
{
  "audio_path": "speaker1/angry_001.wav",
  "label": 2,
  "emotion_text": "angry",
  "speaker_id": "liuchanhg",
  "duration": 1.79,
  "start_time_ms": 0,
  "end_time_ms": 1790
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_path` | `str` | ✓ | 音频文件路径（相对或绝对） |
| `label` | `int` | ✓ | 情感类别 ID |
| `emotion_text` | `str` | | 情感文本标签 |
| `speaker_id` | `str` | | 说话人 ID |
| `duration` | `float` | | 音频时长（秒） |
| `start_time_ms` | `int` | | 片段起始时间（毫秒） |
| `end_time_ms` | `int` | | 片段结束时间（毫秒） |

---

## 7. 使用示例 (Usage Examples)

### 7.1 基础用法

```python
from torch.utils.data import DataLoader
from ser_lib.dataset.spectrogram_dataset import SpectrogramDataset
from ser_lib.dataset.collate import build_collate_fn

# 初始化数据集
dataset = SpectrogramDataset(
    config_path="ser_lib/dataset/configs/casia.yaml",
    split="train"
)

# 构建 DataLoader
collate_fn = build_collate_fn(dataset.config)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

# 迭代训练
for batch, labels in dataloader:
    # batch['x']['melspectrogram']: [B, 1, 80, 300]
    # labels: [B]
    pass
```

### 7.2 获取类别权重

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

labels = dataset.get_labels()
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[l] for l in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
```

---

## 8. 文件结构 (File Structure)

```
ser_lib/dataset/
├── base_dataset.py          # 抽象基类
├── waveform_dataset.py      # 原始波形数据集
├── spectrogram_dataset.py   # 谱图数据集
├── feature_dataset.py       # 多特征数据集
├── collate.py               # 批处理函数工厂
├── augment/
│   ├── builder.py           # 增强流水线构建器
│   ├── time_domain.py       # 时域增强类
│   └── freq_domain.py       # 频域增强类
├── features/
│   ├── builder.py           # 特征提取器工厂
│   ├── spectrogram.py       # 谱图提取器
│   ├── time_domain.py       # 时域特征提取器
│   └── freq_domain.py       # 频域特征提取器
├── configs/
│   ├── casia.yaml           # CASIA 数据集配置
│   └── casia/
│       ├── train.jsonl      # 训练集元数据
│       ├── val.jsonl        # 验证集元数据
│       └── test.jsonl       # 测试集元数据
├── waveform_dataset.yaml    # WaveformDataset 配置模板
├── spectrogram_dataset.yaml # SpectrogramDataset 配置模板
└── feature_dataset.yaml     # FeatureDataset 配置模板
```
