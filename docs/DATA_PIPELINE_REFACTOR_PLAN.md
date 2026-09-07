# SER 数据加载与表示系统重构实施计划

> 文档用途：交给负责编码、测试和审查的智能体作为统一实施依据。
>
> 当前状态：设计阶段。本文档定义目标、边界、接口、迁移顺序和验收标准；实现过程中如需修改公开契约，必须先更新本文档，再修改代码。

## 实施状态（2026-09-07）

数据管线重构的计划范围已经完成首轮实现：

- T0：补全打包配置、pytest 配置、构建产物忽略规则；CI 尚需在远程仓库配置。
- T1：核心类型、异常层级、TensorSpec 和运行时契约已实现。
- T2：Manifest、AudioLoader、Folder/CSV/JSONL/CASIA Importer 已实现。
- T3：Registry、Raw/谱图/MFCC/声学/组合表示、Transform、Pipeline、唯一核心 Dataset 已实现。
- T4：Dynamic/Fixed/Sliding Collator 已实现；旧协议适配层已移除。
- T5：项目尚未公开发布，已决定不保留旧 Dataset、旧配置 Schema 和 tuple 协议。
- T6：ModelSpec、CNN 基线、Trainer、Evaluator、checkpoint、离线推理、滑窗聚合和模型 artifact 已实现。
- T7：确定性 Representation 缓存、损坏自愈、缓存统计和人工性能基准已实现。
- T8：组件目录、兼容性校验和数据集校验能力已实现；桌面服务代码已按仓库边界移除。

移除旧兼容测试并增加新数据生成器测试后，重构完成时的自动化基线为 `38 passed`；移除服务专属代码后当前基线为 `34 passed`。实时推理和远程 CI 属于下一阶段基础库建设，不属于本次数据管线重构的完成条件。

## 1. 背景与重构目标

当前数据模块已经包含配置校验、音频读取、三种数据表示、增强与多种批处理策略，但职责耦合较重：

- `BaseConfigDataset` 同时负责 YAML 加载、路径解析、manifest 读取、音频解码、片段加载、声道转换和重采样。
- `WaveformDataset`、`SpectrogramDataset`、`FeatureDataset` 将“数据集”和“输入表示”绑定在一起。
- Dataset 子类直接理解完整配置结构，难以单独测试和复用。
- `collate.py` 根据 Dataset 类型复制多套批处理分支，代码规模随表示类型增长。
- 特征输出 key 和 tensor layout 缺少稳定契约，模型只能依赖隐含约定。
- 多特征模式强制所有特征具有相同时间长度，不适用于不同 hop length、全局特征或预训练表示。
- 训练、推理、数据预览尚未共享同一套音频预处理和表示流水线。

本次重构的首要目标不是增加功能，而是建立稳定、可测试、可由任意 Python 调用方复用的运行时契约。

### 1.1 必须达成的目标

1. 只保留一个核心 `SERDataset`，Dataset 不再按 waveform、spectrogram、feature 分类。
2. 将数据来源、音频加载、增强、输入表示、批处理完全解耦。
3. 所有训练、验证、测试和离线推理复用相同的 `AudioLoader` 与 `Representation`。
4. 用明确的数据类型描述单样本、批次、tensor layout 和长度。
5. 用注册表扩展 importer、transform、representation 和 collator，避免中央 `if/elif` 持续增长。
6. 项目公开发布前直接移除旧 Dataset API，只保留新的 `ser_lib.data` 公共入口。
7. 为调用方提供可枚举、可校验的组件描述信息。
8. 建立足够测试，使后续模型层不依赖数据层实现细节。

### 1.2 非目标

本轮不负责：

- 实现完整训练器、评估器或任何 UI。
- 一次性加入大量模型或数据集。
- 实现在线多人标注、云端对象存储或分布式训练。
- 自动兼容任意第三方 Python 模型。
- 为不同表示自动猜测模型输入维度。
- 在重构过程中更改已有 CASIA 数据划分内容。

## 2. 当前代码中的已知问题

实施智能体必须先理解并覆盖以下问题。

### 2.1 配置对象使用错误

`BaseConfigDataset` 将 `load_config()` 的 Pydantic 对象保存在 `self.config`，同时生成 `self.config_dict`。三个子类却调用 `self.config.get(...)`。Pydantic `BaseModel` 不应作为普通字典使用。

短期修复应统一访问方式；目标架构中 Dataset 不直接持有完整配置对象。

### 2.2 README 与实现不一致

README 展示了 `EmotionRecognizer`，但离线推理模块尚未实现。重构不得继续建立只存在于文档中的公开接口。

### 2.3 路径解析依赖当前工作目录

配置文件中的相对路径目前可能相对于进程启动目录解析。Notebook、CLI、测试和普通 Python 程序的当前工作目录不同，会造成同一配置在不同入口下表现不同。

目标规则：

- manifest 路径相对于配置文件所在目录解析；
- manifest 内部音频相对路径相对于 manifest 声明的 `root` 解析；
- 数据集根目录内部尽量保存相对路径，外部引用保存规范化绝对路径；
- 不允许依赖 `cwd` 作为隐式业务路径。

### 2.4 特征时间轴被错误统一

当前 `FeatureDataset` 断言全部特征最后一维相同。新设计必须支持：

- 多个共享时间轴的帧级特征；
- 不同时间分辨率的多分支特征；
- 帧级特征与 utterance-level 全局向量同时存在；
- 某些输入没有时间长度。

禁止在 Dataset 中无条件插值或裁剪来“凑齐”长度。只有显式配置的对齐组件可以进行时间对齐。

### 2.5 未实现组件出现在可选配置中

Mixup、RIR、动态噪声等存在可配置但运行时抛 `NotImplementedError` 的情况。新注册表不得发布不可运行组件。实验性组件必须声明状态，并且默认不出现在公开稳定组件列表中。

### 2.6 未知组件被静默跳过

增强 builder 遇到未知类型时可能不报错。所有未知组件、未知参数和不兼容组合必须在任务启动前失败，禁止静默降级。

## 3. 设计原则与硬性约束

以下规则属于实现约束，不是建议。

1. **Dataset 只组装样本。** 它不读取 YAML，不判断表示类型，不理解模型结构。
2. **Representation 决定输入语义。** waveform、Log-Mel、MFCC、F0 等是表示，不是 Dataset 类型。
3. **Collator 根据输入规格工作。** 禁止通过 Dataset 类名判断应该怎样 padding。
4. **模型显式声明输入要求。** 禁止调用方或训练器猜测 tensor shape。
5. **训练和推理共享预处理。** 模型 artifact 必须携带推理所需表示配置。
6. **验证优先于运行。** 配置错误、key 不匹配和 layout 不匹配必须在训练启动前发现。
7. **路径解析确定且可复现。** 不依赖当前工作目录。
8. **随机性可控制。** 增强在固定 seed 下必须可复现，验证和测试默认禁用随机增强。
9. **导入不复制大文件。** 默认引用原始音频；复制到目标目录必须由调用方显式选择。
10. **公共接口不泄露内部配置细节。** 外部调用方只依赖公开类型、descriptor 和构建函数。
11. **不反序列化不可信 pickle。** 模型加载优先 `safetensors`、ONNX 或受控 `state_dict`。
12. **错误必须可定位。** 异常至少包含记录 ID、解析后路径、组件名称和失败阶段。

## 4. 目标目录结构

建议将现有 `ser_lib/dataset` 逐步迁移为 `ser_lib/data`。迁移期间两者可以共存。

```text
ser_lib/
├── data/
│   ├── __init__.py
│   ├── types.py
│   ├── errors.py
│   ├── manifest.py
│   ├── audio.py
│   ├── dataset.py
│   ├── pipeline.py
│   ├── collate.py
│   ├── registry.py
│   ├── cache.py
│   ├── validation.py
│   ├── importers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── folder.py
│   │   ├── csv_importer.py
│   │   └── casia.py
│   ├── representations/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── waveform.py
│   │   ├── spectral.py
│   │   ├── acoustic.py
│   │   └── composite.py
│   └── transforms/
│       ├── __init__.py
│       ├── base.py
│       ├── waveform.py
│       └── feature.py
├── models/
│   ├── base.py
│   └── registry.py
└── artifacts/
    ├── manifest.py
    ├── loader.py
    └── exporter.py
```

仓库根目录中存放真实语料的 `data/` 后续建议更名为 `datasets/`；本次不要自动移动用户数据。

## 5. 核心公开类型

第一阶段必须先冻结这些类型，再实施其他模块。类型字段如需修改，必须同步更新测试和本文档。

### 5.1 `AudioRecord`

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class AudioRecord:
    uid: str
    audio_path: Path
    label: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    speaker_id: str | None = None
    sample_rate_hint: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

约束：

- `uid` 在一个 manifest 中必须唯一。
- `audio_path` 可以是相对路径，解析由 Manifest/Resolver 完成。
- `start_ms >= 0`。
- 同时存在时必须满足 `end_ms > start_ms`。
- `label=None` 用于无标签推理数据。
- `metadata` 不得被下游组件原地修改。
- 不在核心类型中固化性别、语种等无限增长的业务字段，它们进入 `metadata`。

### 5.2 `AudioData`

```python
@dataclass(frozen=True, slots=True)
class AudioData:
    waveform: torch.Tensor       # [C, T]
    sample_rate: int
    source_path: Path
    original_sample_rate: int
    num_frames: int
```

约束：

- Loader 输出始终是二维 `[C, T]`。
- 单声道策略开启时 `C == 1`，仍然保留 channel 维。
- 默认 dtype 为 `torch.float32`。
- Loader 不负责 `squeeze()`，避免单样本维度随输入变化。
- 空音频和零帧片段必须抛业务异常。

### 5.3 `TensorSpec`

```python
@dataclass(frozen=True, slots=True)
class TensorSpec:
    layout: str
    dtype: torch.dtype
    feature_dim: int | None = None
    time_axis: int | None = None
    pad_value: float = 0.0
```

第一版允许的 layout：

| Layout | 单样本含义 | 批次含义 |
|---|---|---|
| `T` | 原始波形 `[T]` | `[B,T]` |
| `FT` | 频率/特征优先 `[F,T]` | `[B,F,T]` |
| `TD` | 时间优先 `[T,D]` | `[B,T,D]` |
| `D` | 全局向量 `[D]` | `[B,D]` |
| `CFT` | 多通道谱图 `[C,F,T]` | `[B,C,F,T]` |

注意：

- `B` 只存在于批次，不写入单样本 `layout`。
- 第一版不实现通用 layout 解析器，只支持白名单，并在验证函数中显式映射。
- `time_axis=None` 表示非时序输入，例如 `D`。
- 不允许通过 tensor rank 猜 layout，因为 `[80, 300]` 既可能是 `FT` 也可能是 `TD`。

### 5.4 `RepresentationOutput`

```python
@dataclass(frozen=True, slots=True)
class RepresentationOutput:
    inputs: dict[str, torch.Tensor]
    lengths: dict[str, int]
```

约束：

- 单一波形表示统一使用 key `waveform`。
- 单一声学表示统一使用 key `features`。
- 多分支表示使用有语义的稳定 key，如 `mel`、`prosody`。
- 只有时序输入需要出现在 `lengths`。
- `lengths` 中的 key 必须属于 `inputs`。
- 每个 tensor 必须满足 Representation 对外声明的 `TensorSpec`。

### 5.5 `SERSample`

```python
@dataclass(frozen=True, slots=True)
class SERSample:
    uid: str
    inputs: dict[str, torch.Tensor]
    lengths: dict[str, int]
    label: int | None
    metadata: dict[str, Any]
```

### 5.6 `SERBatch`

优先使用 dataclass，而不是长期依赖无约束嵌套字典：

```python
@dataclass(frozen=True, slots=True)
class SERBatch:
    inputs: dict[str, torch.Tensor]
    lengths: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    labels: torch.Tensor | None
    uids: list[str]
    metadata: list[dict[str, Any]]
    window_map: torch.Tensor | None = None
```

约束：

- 分类标签 dtype 为 `torch.long`。
- 无标签样本组成的推理 batch，`labels=None`。
- 一个 batch 内不允许部分样本有标签、部分没有；Collator 必须报错。
- mask 使用 `True` 表示有效位置，`False` 表示 padding。
- `window_map[i]` 表示第 `i` 个滑窗来自原始 batch 的哪个样本。
- 不再通过 `original_labels` 和 `window_counts` 间接推导滑窗映射。

## 6. Manifest 与数据导入

### 6.1 标准 manifest 格式

内部标准格式使用 JSONL，一行一个样本：

```json
{"uid":"casia-000001","audio_path":"neutral/001.wav","label":0,"speaker_id":"speaker-a","metadata":{"language":"zh"}}
```

片段记录：

```json
{"uid":"meeting-001-0003","audio_path":"meeting-001.wav","label":2,"start_ms":15320,"end_ms":18910}
```

Manifest 元信息单独保存，例如 `dataset.yaml`：

```yaml
schema_version: 1
dataset_id: casia
root: D:/datasets/CASIA
splits:
  train: train.jsonl
  val: val.jsonl
  test: test.jsonl
labels:
  0: {en: neutral, zh: 平静}
  1: {en: happy, zh: 高兴}
```

### 6.2 Manifest 责任

- 读取和写入 JSONL。
- 校验必填字段、UID 唯一性、label 范围和片段时间。
- 将音频路径解析为确定路径。
- 支持迭代和按 split 获取记录。
- 提供轻量统计，不执行音频解码。

### 6.3 Importer 责任

Importer 将外部数据格式转成标准 manifest：

```python
class DatasetImporter(Protocol):
    descriptor: ComponentDescriptor

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        ...

    def convert(self, source: Path, destination: Path, config: Mapping[str, Any]) -> DatasetManifest:
        ...
```

必须区分 `scan()` 和 `convert()`：调用方先检查解析结果、标签映射和错误，再决定是否写入目标目录。

首批 importer：

1. `folder`：从目录和文件名规则导入。
2. `csv`：映射音频路径列、标签列和可选元数据列。
3. `jsonl`：验证已有标准或近似标准 manifest。
4. `casia`：将当前 CASIA 脚本迁移为适配器。

不要在核心 Dataset 中加入 CASIA/RAVDESS 特殊逻辑。

### 6.4 路径安全注意事项

- 对用户选择的目录使用 `Path.resolve()` 后记录。
- 引用模式不复制音频；数据根目录被移动时应报告外部引用失效。
- 复制模式必须防止目标覆盖、路径穿越和同名冲突。
- ZIP 导入必须拒绝解压到目标目录之外的成员路径。
- 文件扫描应支持取消、进度和错误汇总，不能因一个损坏文件终止全部扫描。
- 大目录扫描不得一次将所有文件内容或音频加载进内存。

## 7. AudioLoader 设计

### 7.1 建议接口

```python
@dataclass(frozen=True)
class AudioLoaderConfig:
    target_sample_rate: int = 16000
    mono: bool = True
    normalize_peak: bool = False
    backend: str = "torchaudio"

class AudioLoader:
    def __init__(self, config: AudioLoaderConfig):
        ...

    def load(self, record: AudioRecord) -> AudioData:
        ...
```

### 7.2 执行顺序

1. 解析并验证最终路径。
2. 读取音频元信息。
3. 根据原采样率将毫秒片段转换为 frame offset。
4. 只读取目标片段，避免加载整段长音频。
5. 校验读取结果非空且有限值。
6. 根据策略转换声道。
7. 重采样至目标采样率。
8. 可选执行确定性的 loader-level 归一化。
9. 返回 `[C,T]` 的 float32 tensor。

### 7.3 边界条件

- `end_ms` 超出文件长度：第一版允许截断，但应产生可观测 warning；严格模式可报错。
- `start_ms` 超出音频长度：报 `InvalidAudioSegmentError`。
- 极短音频：Loader 可以返回，最小长度限制由 pipeline validation 决定。
- NaN/Inf：报错，不自动替换为零。
- 多声道转单声道：默认求均值；以后可以扩展 channel selection。
- 重采样器缓存 key 至少包含 `(original_sr, target_sr, dtype, device)`。
- Dataset worker 间不要共享不可 pickle 的全局 decoder 状态。

### 7.4 异常层级

```text
SERDataError
├── ManifestError
├── AudioNotFoundError
├── AudioDecodeError
├── InvalidAudioSegmentError
├── RepresentationError
├── CollationError
└── CompatibilityError
```

异常消息必须包含 `uid` 和解析后的音频路径；保留原始异常作为 `__cause__`。

## 8. Transform 与随机增强

### 8.1 分层

只保留明确阶段：

```text
waveform transforms
    ↓
representation
    ↓
feature transforms
    ↓
collation
    ↓
batch transforms
```

- 波形级：噪声、音量、时间偏移、pitch shift。
- 表示级：Log-Mel、MFCC、F0。
- 特征级：SpecAugment、标准化。
- 批次级：Mixup，仅在形成 batch 后执行。

禁止把 Mixup 放在 Dataset 的单样本 pipeline 中。

### 8.2 概率包装器

组件不应各自重复实现 `p` 判断，使用统一包装：

```python
class RandomApply(nn.Module):
    def __init__(self, transform: nn.Module, probability: float):
        ...
```

概率必须校验在 `[0,1]`。随机源应能接受 worker seed，保证 DataLoader 多进程下可复现。

### 8.3 训练与评估隔离

- `split=train` 默认允许随机增强。
- `val/test/predict` 默认禁用随机增强。
- 如果用户显式要求测试时增强（TTA），应作为独立推理策略，而不是复用训练增强开关。
- 表示本身必须是确定性的；随机操作属于 transform。

### 8.4 未实现功能处理

- 未完成的 RIR、动态噪声、Mixup 不得注册为 stable。
- descriptor 中使用 `status: stable | experimental | unavailable`。
- 普通组件列表只返回 stable。
- experimental 功能必须有单独测试和明确警告。

## 9. Representation 系统

### 9.1 基础接口

```python
class Representation(nn.Module, ABC):
    descriptor: ComponentDescriptor

    @property
    @abstractmethod
    def output_specs(self) -> dict[str, TensorSpec]:
        ...

    @abstractmethod
    def forward(self, audio: AudioData) -> RepresentationOutput:
        ...
```

Representation 不读取文件、不处理 label、不知道 split。

### 9.2 第一阶段表示

#### RawWaveform

- 输入：`AudioData.waveform [1,T]`。
- 输出：`inputs["waveform"] [T]`。
- layout：`T`。
- length：`{"waveform": T}`。

#### LogMel

- 输入：`[1,T]`。
- 输出：`inputs["features"] [F,Tm]`。
- layout：`FT`。
- length：`{"features": Tm}`。
- 参数至少包含 sample rate、n_fft、win_length、hop_length、n_mels、f_min、f_max、power、top_db。
- Representation 必须验证配置 sample rate 与 Loader 输出一致，不得静默以不同采样率计算。

#### MFCC

- 输出同样使用 `features`，layout 为 `FT`。
- Mel 参数转换逻辑封装在组件内部，不暴露给 Dataset。

### 9.3 第二阶段表示

- `AcousticFeatures`：F0、RMS、ZCR 等。
- `CompositeRepresentation`：组合多个子表示。
- `PretrainedEmbedding`：HuBERT/Wav2Vec2 等 encoder 输出。
- `CachedRepresentation`：装饰器，不改变输出协议。

### 9.4 多特征对齐

默认不对齐不同输出。`CompositeRepresentation` 可以返回：

```python
inputs = {
    "mel": mel_tensor,              # [F,T1]
    "prosody": prosody_tensor,      # [T2,D]
    "global": global_tensor,        # [D2]
}
lengths = {"mel": T1, "prosody": T2}
```

只有用户显式选择 `TemporalAligner` 时才执行：

- `reference`: 参考输入 key；
- `strategy`: `crop | pad | interpolate`；
- `target_axis`: 要对齐的时间轴；
- 是否允许插值必须由特征语义决定，类别型/离散表示不能默认线性插值。

第一版可以不实现自动对齐，但必须保留不等长多输入的数据结构。

## 10. `SERDataset` 与 Pipeline

### 10.1 Dataset 接口

```python
class SERDataset(torch.utils.data.Dataset[SERSample]):
    def __init__(
        self,
        records: Sequence[AudioRecord],
        audio_loader: AudioLoader,
        pipeline: SamplePipeline,
    ) -> None:
        ...
```

`__getitem__` 只做：

```text
record = records[index]
audio = audio_loader.load(record)
sample = pipeline(audio, record)
validate sample contract
return sample
```

### 10.2 Pipeline 接口

```python
class SamplePipeline:
    def __init__(
        self,
        waveform_transforms: nn.Module,
        representation: Representation,
        feature_transforms: nn.Module,
    ): ...

    @property
    def output_specs(self) -> dict[str, TensorSpec]: ...

    def __call__(self, audio: AudioData, record: AudioRecord) -> SERSample: ...
```

### 10.3 性能注意事项

- 不在 Dataset 初始化时解码全部音频。
- 不在每个 `__getitem__` 重建 torchaudio transform。
- Representation 与 transform 在构造时创建一次。
- 不将 GPU tensor 存在 Dataset；DataLoader worker 只处理 CPU tensor。
- 送入 GPU 由训练循环负责，并可使用 `pin_memory`。
- Dataset 对象应可被 DataLoader worker pickle。
- 避免在 worker 内打开长期不关闭的普通文件句柄。

## 11. 通用 Collator

### 11.1 目标

用输入规格取代 waveform/spectrogram/feature 三套分支。

```python
class SERCollator:
    def __init__(
        self,
        specs: dict[str, TensorSpec],
        strategy: CollateStrategy,
    ): ...

    def __call__(self, samples: list[SERSample]) -> SERBatch: ...
```

### 11.2 第一阶段策略

#### Dynamic padding

- 每个时序 key 独立计算该 batch 最大长度。
- 沿 `TensorSpec.time_axis` padding。
- 生成对应 key 的 bool mask。
- `D` 类型直接 stack，不生成 length 或 mask。

#### Fixed length

- 配置应支持每个 key 的最大长度，而不是只有一个 `max_frames`。
- 超长截断，短样本 padding。
- 输出 length 是截断后的有效长度。

配置示例：

```yaml
batching:
  type: fixed
  max_lengths:
    features: 300
```

#### Sliding window

- 第一版只支持一个主时序输入。
- 多输入滑窗必须先定义同步切窗语义，不允许简单对每个 key 独立切窗。
- 输出 `window_map`，保留窗口到原始样本的明确映射。
- 最后一个不足窗口长度的片段需要 padding，length 为实际有效长度。
- 当输入恰好等于窗口长度时只产生一个窗口。
- 短输入也至少产生一个窗口。

### 11.3 Batch transform

Mixup 在 Collator 基础结果之后执行：

```text
SERCollator(samples)
    ↓
SERBatch
    ↓
BatchTransformPipeline
```

Mixup 会产生软标签，届时 `SERBatch.labels` 的协议需要扩展。实现 Mixup 前必须先确定：

- hard label 和 soft label 的类型表示；
- loss 如何选择；
- metrics 如何使用原始标签；
- sliding window 与 Mixup 是否允许组合。

在这些契约未确定前，Mixup 保持 unavailable，不要留下运行时占位分支。

## 12. 配置与组件注册表

### 12.1 顶层配置保持强类型

建议：

```python
class DataConfig(BaseModel):
    manifest: Path
    audio: AudioLoaderConfig
    representation: ComponentConfig
    waveform_transforms: list[ComponentConfig] = []
    feature_transforms: list[ComponentConfig] = []
    batching: BatchingConfig

class ComponentConfig(BaseModel):
    type: str
    params: dict[str, Any] = {}
```

顶层负责结构校验，每个组件负责自己的参数 Schema。不要继续让一个中央配置文件枚举所有未来参数。

### 12.2 注册表接口

```python
registry.register(
    namespace="representation",
    name="log_mel",
    factory=LogMelRepresentation,
    config_model=LogMelConfig,
    descriptor=ComponentDescriptor(...),
)
```

注册时必须检查：

- `(namespace, name)` 唯一；
- descriptor ID 与注册名称一致；
- config model 可实例化；
- stable 组件具有实现和测试；
- 重复注册默认报错，不能后者静默覆盖前者。

### 12.3 `ComponentDescriptor`

Python API 和 CLI 可通过 descriptor 枚举组件及其参数：

```python
@dataclass(frozen=True)
class ComponentDescriptor:
    id: str
    display_name: str
    category: str
    version: str
    status: str
    description: str
    config_schema: dict[str, Any]
    input_specs: dict[str, TensorSpec] | None
    output_specs: dict[str, TensorSpec] | None
```

Pydantic 配置类的 JSON Schema 是公开的机器可读参数描述，但最终配置仍必须由注册表校验。

## 13. 模型兼容性契约

虽然本轮重点是数据模块，但必须为模型层留下明确边界。

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    required_inputs: dict[str, TensorSpec]
    supports_masks: bool
    supports_variable_length: bool
    num_classes: int | None
```

实现 `validate_compatibility(representation_specs, model_spec, batching_config)`，至少校验：

1. 所有 required input key 存在。
2. layout 完全匹配；第一版不自动转置。
3. 固定 feature dimension 匹配。
4. 可变长度输入是否有模型支持的 mask。
5. fixed-only 模型是否配置固定长度 Collator。
6. 数据集 label 数量与模型输出类别一致。
7. sliding window 是否有推理聚合策略。

兼容性错误必须在创建训练任务时返回，不得等到第一个 forward 才通过 shape error 暴露。

## 14. 特征缓存

缓存属于 Representation 装饰器，不属于独立 Dataset。

### 14.1 缓存 key

至少包含：

```text
schema version
+ 音频规范化路径
+ 文件大小
+ 文件修改时间或内容哈希
+ start/end segment
+ target sample rate
+ representation ID 和版本
+ representation 完整参数
+ 确定性预处理参数
+ ser_lib 兼容版本
```

开发阶段可使用 `size + mtime_ns` 提高速度；可复现发布或模型 artifact 构建应支持内容哈希。

### 14.2 缓存规则

- 验证、测试和确定性推理表示可以缓存。
- 随机波形增强之后的输出默认不得缓存。
- 缓存写入使用临时文件加原子替换，防止并发 worker 写出半文件。
- 缓存损坏时删除单个条目并重新计算，不清空整个缓存。
- 缓存必须记录 dtype、shape 和 spec，加载后重新校验。
- 公共 API 和 CLI 应能报告缓存大小并清理指定数据集或表示的缓存。

## 15. 配置示例

### 15.1 Log-Mel 训练

```yaml
schema_version: 1
data:
  manifest: ./dataset.yaml
  audio:
    target_sample_rate: 16000
    mono: true
  representation:
    type: log_mel
    params:
      n_fft: 1024
      win_length: 1024
      hop_length: 256
      n_mels: 80
      f_min: 0.0
      f_max: 8000.0
      power: 2.0
      top_db: 80.0
  waveform_transforms:
    - type: gaussian_noise
      probability: 0.3
      params:
        snr_db: 15.0
  feature_transforms:
    - type: spec_masking
      probability: 0.5
      params:
        time_mask_param: 30
        freq_mask_param: 15
  batching:
    type: dynamic
```

### 15.2 原始波形训练

```yaml
data:
  manifest: ./dataset.yaml
  audio:
    target_sample_rate: 16000
    mono: true
  representation:
    type: waveform
    params: {}
  batching:
    type: dynamic
```

### 15.3 多分支输入

```yaml
data:
  representation:
    type: composite
    params:
      outputs:
        mel:
          type: log_mel
          params:
            n_mels: 80
            hop_length: 320
        prosody:
          type: acoustic_features
          params:
            features: [f0, rms, zcr]
            hop_length: 320
  batching:
    type: dynamic
```

第一阶段不要求实现第三个示例，但设计不能阻止它。

## 16. 旧接口移除决策

项目尚未正式发布，没有需要维持的外部 API，因此不保留兼容包装器和配置迁移器。

已经删除：

- `ser_lib.dataset` 包；
- `WaveformDataset`、`SpectrogramDataset`、`FeatureDataset`；
- 旧 `build_collate_fn` 和 tuple 样本协议；
- 旧的集中式 `config_schema.py`；
- 三类旧 YAML 模板和迁移测试。

数据预处理工具直接生成标准 `dataset.yaml`、JSONL 和新的 `DataConfig` 文件。所有代码、测试和教程只能使用 `ser_lib.data`。

## 17. 分阶段实施任务

编码智能体应按依赖顺序执行。不同智能体并行工作时，只能领取没有未完成接口依赖的任务。

### 阶段 0：基线冻结与修复

#### T0.1 建立测试运行环境

- 补全 `pyproject.toml`。
- 定义主依赖与 `test` optional dependency。
- 确保 `pip install -e .[test]` 可用。
- 配置 pytest。
- 加入最小 CI：支持的最低 Python 版本和一个主版本。

验收：全新虚拟环境能够安装、import 并运行测试。

#### T0.2 冻结当前行为

- 为现有三类 Dataset 和 collate 添加 characterization tests。
- 测试不应把已知 bug 固化为正确行为；已知 bug单独写失败测试后修复。
- 创建少量临时 WAV，不依赖用户本地 CASIA 数据。

#### T0.3 修复当前配置访问错误

- 将三个 Dataset 的 `self.config.get()` 修复为一致访问方式。
- 不在此任务中进行结构重写。

验收：三种现有 Dataset 均可成功实例化并读取一条测试音频。

### 阶段 1：冻结新核心类型

#### T1.1 实现 types 与 errors

- 实现本文第 5 节数据类型。
- 实现异常层级。
- 实现 tensor/spec 验证工具。

#### T1.2 定义序列化边界

- 明确 dataclass 到 JSON-safe descriptor 的转换。
- `torch.dtype` 序列化为稳定字符串，如 `float32`。
- Path 序列化由配置/manifest 层负责，运行时类型仍用 `Path`。

验收：类型单元测试覆盖合法、缺 key、错 layout、错 length、部分标签等情况。

### 阶段 2：Manifest 与 AudioLoader

#### T2.1 实现 Manifest

- JSONL 读取/写入。
- split 和 dataset metadata。
- UID、label、片段校验。
- 相对路径确定性解析。

#### T2.2 实现 AudioLoader

- 整段与片段加载。
- 单/多声道处理。
- 重采样。
- 空音频和损坏音频异常。
- resampler 缓存。

#### T2.3 实现导入预览

- folder、CSV、JSONL importer。
- scan/convert 分离。
- 错误按条目收集。

验收：从临时目录导入、写 manifest、重新加载并解码的端到端测试通过。

### 阶段 3：Representation 与 Pipeline

#### T3.1 建立注册表和 descriptor

- 实现命名空间隔离。
- 重复注册报错。
- 每组件独立 Pydantic 配置。
- 枚举 stable descriptor。

#### T3.2 实现首批表示

- RawWaveform。
- LogMel。
- MFCC。
- 输出统一 key 和 spec。

#### T3.3 迁移稳定增强

- Normalize。
- Gaussian noise。
- Time shift。
- Volume scale。
- Spec masking。
- 暂不迁移未实现组件。

#### T3.4 实现 SamplePipeline 与 SERDataset

- 串联 Loader、transform、representation。
- 输出 `SERSample`。
- 运行时 contract validation 可通过 debug/strict 开关控制；测试和开发默认开启。

验收：相同音频和配置在 Dataset 与直接 pipeline 调用中输出一致。

### 阶段 4：通用 Collator

#### T4.1 Dynamic padding

- 支持 `T/FT/TD/D/CFT`。
- 多 key 独立 padding。
- 正确生成 length 与 mask。

#### T4.2 Fixed length

- 支持按 key 配置长度。
- 验证截断后的 length。

#### T4.3 Sliding window

- 第一版仅单主时序 key。
- 输出 window map。
- 添加窗口聚合所需元数据。

#### T4.4 旧 collate 适配

- 现有测试在兼容层继续通过。
- 新测试只依赖 `SERBatch`。

验收：不再根据 Dataset 类型分派 padding 实现；核心 Collator 无 waveform/spectrogram/feature 分支。

### 阶段 5：旧系统迁移

#### T5.1 删除旧接口

- 删除旧三类 Dataset、旧 collate、旧 Schema、迁移器和配置模板。
- 数据生成脚本直接输出新 manifest 与 `DataConfig`。
- 测试和文档不得导入 `ser_lib.dataset`。

#### T5.2 更新教程与 README

- 快速开始示例必须使用 `ser_lib.data`。
- 新教程使用新 API。
- 不留下两套互相矛盾的推荐方式。

验收：仓库业务代码和测试中不存在 `ser_lib.dataset` 引用。

### 阶段 6：模型与训练衔接

#### T6.1 ModelSpec 与兼容性检查

- 定义模型输入规格。
- 在训练启动前校验表示、batching 和模型。

#### T6.2 Trainer 接受 SERBatch

- 设备搬运递归处理 `inputs/lengths/masks/labels`。
- Trainer 不根据表示类型分支。

#### T6.3 推理共享 pipeline

- 推理 artifact 加载同一表示配置。
- 单文件推理用 `AudioRecord(label=None)`。
- 禁用训练随机增强。

验收：训练保存的预处理配置可以直接恢复并完成离线推理。

### 阶段 7：缓存与性能

- 实现确定性表示缓存。
- benchmark `num_workers=0/2/4`。
- 测量解码、重采样、表示计算和 collate 时间。
- 优化必须基于 profile，不凭感觉加入复杂并发。

验收：缓存命中输出与未缓存输出一致；并发写入不产生损坏文件。

### 阶段 8：公共发现与校验接口

- 枚举 importer、representation、transform、model descriptor。
- 数据集扫描/验证支持进度和取消。
- 训练任务配置返回结构化兼容性错误。
- 所有长操作通过与界面无关的 Python callback/protocol 上报进度。

## 18. 测试计划

### 18.1 单元测试

#### Manifest

- JSONL 单条、多条、空文件。
- 重复 UID。
- label 越界。
- 相对和绝对路径。
- 中文路径、空格路径。
- 合法和非法片段时间。

#### AudioLoader

- 8k、16k、44.1k 重采样。
- mono/stereo。
- 整段和片段。
- 超短、空、损坏、不存在音频。
- 输出 dtype、shape、有限值。

#### Representation

- 每种表示的输出 key、layout、shape、length。
- 不同音频长度。
- 参数边界。
- sample rate 不匹配。
- 相同输入下确定性。

#### Transform

- `p=0` 不改变输入。
- `p=1` 必定执行。
- 固定 seed 可复现。
- 不原地破坏输入，除非接口明确声明。
- 输出 shape 是否允许变化必须单独测试。

#### Collator

- `T/FT/TD/D/CFT`。
- 单样本和多样本。
- 不同长度、多 key 不同长度。
- dynamic/fixed/sliding。
- mask 语义。
- 部分 label 报错。
- 输入 key 不一致报错。

### 18.2 集成测试

1. Importer → manifest → loader → pipeline → dataset → collator。
2. 旧 YAML → migration → 新 DataConfig → 新 Dataset。
3. Dataset 与离线推理预处理结果一致。
4. 多 worker DataLoader 可迭代多个 epoch。
5. 缓存未命中 → 写入 → 命中 → 参数变化后重新计算。

### 18.3 回归测试

- 新 Collator 测试覆盖旧测试具有价值的动态、固定和滑窗行为。
- 数据处理器生成的新 manifest 必须保持 label mapping 和 split 隔离。
- 对固定测试 WAV 保存关键输出 shape 和合理数值范围，不建议保存依赖库版本敏感的完整浮点数组。

### 18.4 性能测试

性能测试不应成为普通单元测试的硬门槛，但需记录基线：

- 每秒解码音频时长。
- 每秒生成 Log-Mel 的样本数。
- DataLoader 首 batch 和稳定状态耗时。
- 峰值内存。
- cache hit/miss 性能。

## 19. 代码质量要求

- 所有公开类型和函数有 docstring、类型标注和明确 shape 说明。
- 使用 `Path` 处理路径，不在核心逻辑中手工拼接字符串。
- 不使用 `assert` 校验用户输入；`assert` 在优化模式可能被移除。业务校验抛明确异常。
- 不捕获裸 `Exception` 后静默继续；Importer 批量扫描可以收集错误，但必须保留原因。
- 不在库代码中打印；使用 logging，并允许调用方接收结构化事件。
- 不在 Dataset worker 中执行外部持久化写操作。
- 配置 model 默认 `extra="forbid"`，防止拼错参数被忽略。
- 不把 torch module、打开的文件句柄或锁对象写入配置序列化结果。
- 避免在模块 import 阶段扫描文件、下载模型或初始化 CUDA。
- CPU 是数据管线默认设备；GPU 搬运属于训练/推理执行层。
- 新增公开 API 必须从对应包的 `__init__.py` 导出并记录。

## 20. 基础库集成约束

- 数据模块只暴露 Python API、配置 schema 和组件 descriptor，不实现 HTTP 服务。
- 扫描、导入和缓存生成等长操作使用可选 callback/cancellation token。
- 调用方负责选择目标目录和持久化策略；数据模块负责路径校验与原子写入。
- 组件配置无论由谁生成，都必须由基础库重新校验。
- 独立应用仓库不得依赖 `ser_lib` 私有模块，跨仓库集成只使用公开 API 或 CLI。

## 21. 多智能体编码协作建议

建议将工作拆给不同智能体，但必须按接口依赖合并：

| 工作包 | 内容 | 前置依赖 |
|---|---|---|
| A | pyproject、测试环境、基线测试 | 无 |
| B | types、errors、spec validation | A |
| C | manifest、importers | B |
| D | AudioLoader | B |
| E | registry、representation | B、D |
| F | transforms、pipeline、SERDataset | D、E |
| G | 通用 Collator | B、F |
| H | 删除旧 API、更新数据生成器与文档 | F、G |
| I | ModelSpec 与兼容检查 | B、E、G |
| J | 文档、教程与端到端验证 | H、I |

协作规则：

1. B 包先合并，其他智能体不得各自定义不同版本的 `SERSample` 或 `TensorSpec`。
2. 每个工作包只修改自己拥有的模块；跨模块接口变化先更新设计文档。
3. 不允许两个智能体同时机械重写 `collate.py`。
4. 每个工作包必须同时提交测试，不接受“后续补测试”。
5. 合并前运行全量测试，不只运行本工作包测试。
6. 禁止重新引入旧 tuple 返回协议或按表示类型拆分 Dataset。
7. 如果实现发现本文档契约不可行，应提交最小设计变更说明，包括影响范围和迁移方案。

## 22. 完成定义（Definition of Done）

数据模块重构只有同时满足以下条件才算完成：

- 新代码只有一个核心 `SERDataset`。
- Dataset 不接收 YAML 路径，不读取完整配置。
- waveform、Log-Mel、MFCC 通过 Representation 切换。
- 训练与推理使用同一 AudioLoader 和 Representation。
- Collator 不根据 Dataset 类型分支。
- tensor key、layout、length、mask 均有明确规格与测试。
- 相对路径不依赖当前工作目录。
- 未知组件和未知参数会在运行前报错。
- 仓库中不存在旧三种 Dataset、旧 tuple 协议或旧配置 Schema。
- CASIA 处理器直接生成可由新流水线加载的标准 manifest 与配置。
- 可在干净环境通过 `pip install -e .[test]` 安装。
- 全量单元和集成测试通过。
- README 中所有未标记为伪代码的示例可执行。
- Python 调用方能通过 descriptor 枚举表示与参数，并在运行前执行兼容性检查。

## 23. 首个可交付里程碑

为控制风险，第一个里程碑只要求打通以下最小链路：

```text
临时 WAV/标准 JSONL
    ↓
Manifest
    ↓
AudioLoader（mono + 16 kHz）
    ↓
SERDataset
    ↓
RawWaveform 或 LogMel
    ↓
Dynamic SERCollator
    ↓
SERBatch
```

该里程碑必须支持：

- 有标签训练数据；
- 无标签推理数据；
- 不同长度音频；
- 正确 mask；
- 多 worker DataLoader；
- Windows 中文和空格路径；
- 完整自动化测试。

第一里程碑完成前，不应实现 CompositeRepresentation、特征缓存、Mixup 或大量新 importer。先验证核心契约，再扩展能力。

## 24. 最终审查清单

代码审查者在每个阶段至少确认：

- [ ] 是否把新职责重新塞回 Dataset。
- [ ] 是否出现根据类名或 tensor rank 猜输入语义的逻辑。
- [ ] 是否有依赖当前工作目录的路径。
- [ ] 是否静默忽略未知配置。
- [ ] 是否在验证/测试阶段意外启用随机增强。
- [ ] 是否在 Dataset worker 中使用 GPU 或数据库连接。
- [ ] 是否破坏 `[C,T]`、`FT`、`TD` 等 shape 约定。
- [ ] 是否把多特征不等长错误地强制裁剪。
- [ ] 是否为兼容旧代码污染新核心接口。
- [ ] 是否存在配置声明可用、实际抛 `NotImplementedError` 的组件。
- [ ] 是否有错误信息缺少 uid、路径或组件上下文。
- [ ] 是否新增实现但没有正常、边界和失败路径测试。
- [ ] 是否让调用方重复实现基础库已经提供的兼容性校验。

---

本文档的核心决策是：**数据集只描述样本集合，Representation 描述输入形式，TensorSpec 描述形状契约，Collator 根据规格批处理。** 所有后续模型、训练和推理能力都应建立在这四个边界之上。
