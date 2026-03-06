# SER 数据集模块 (Data Engine)

本模块作为整个语音情感识别项目的数据基座，全权负责音频读取、多模态特征并行提取、变长序列对齐以及多级数据增强等核心任务。

## 目录结构

```text
ser_lib/dataset/
├── base_dataset.py       # 核心配置驱动的数据集类 (`BaseConfigDataset`)
├── collate.py            # 长度重组对齐与批处理闭包构建工厂 (`build_collate_fn`)
├── template_dataset.yaml # 标杆型 YAML 配置模板，控制从采样率、特征提取到策略映射的所有底层变量
├── augment/              # 数据增强核心算子包
│   ├── builder.py        # 增强流水线构建工厂
│   ├── time_domain.py    # 1D 波形信号变换模块 (加噪, 音高, 推移, 混响)
│   └── freq_domain.py    # 2D 频谱增强模块 (掩盖, 滤波, 变形等 SOTA 策略)
├── features/             # 声学特征提取核心包
│   ├── builder.py        # 特征提取构建工厂
│   ├── time_domain.py    # 1D 核心时域韵律及音质特征 (F0, RMS, ZCR, Jitter/Shimmer 等)
│   └── freq_domain.py    # 2D 频谱及高阶动态特征 (MFCC, 谱质心, 滚降, 平坦度等)
└── README.md             # 架构文档及 API 规范说明（本文档）
```

---

## 核心类：`BaseConfigDataset`

继承自 `torch.utils.data.Dataset` 的主体类，实现单样本的独立闭环管理。

### 属性 (Attributes)

- `split` (`str`): 数据集划分 ('train', 'val', 'test')。
- `config` (`dict`): 解析后的总 YAML 配置字典。
- `target_sr` (`int`): 全局目标采样率，音频加载时自动 Resample 依据。
- `data_root_dir` (`str` | `None`): 配置中定义的音频物理根路径前缀，用于拼接相对路径。
- `data_list` (`List[dict]`): 解析完成的核心元数据列表，每个元素对应一个语音切片（参见 `_load_data_list` 格式）。
- `id2label` (`dict`): 从配置文件初始化的分类映射对象，供输出日志和 UI 图例使用。具体结构如下：
  ```python
  {
      0: {"en": "Neutral", "zh": "平静"},
      1: {"en": "Happy", "zh": "高兴"},
      # ...
  }
  ```
- `wave_transform` (`Optional[Callable]`): 基于函数的波形级增强流程（第1级增强，来自 `augment/time_domain.py`）。
- `adv_wave_transform` (`Optional[Callable]`): 保留给复杂噪声注入操作的波形增强接口（第1高级增强，来自 `augment/time_domain.py`）。
- `spec_transform` (`Optional[torch.nn.Module]`): `torchaudio.transforms` 实现的频谱增强容器（如 SpecAugment，第2级增强，来自 `augment/freq_domain.py`）。
- `feature_extractors` (`torch.nn.ModuleDict`): 核心特征提取器映射字典，能够根据配置文件反射组装多种类型的提取管线。
  
### 核心方法 (Methods)

#### `__init__(self, config_path: str, split: str = 'train')`
初始化数据并解析所有计算算子。
- **输入**:
  - `config_path` (`str`): `yaml` 配置文件的物理绝对或相对路径。
  - `split` (`str`): 取值限定为 `'train'`, `'val'`, 甚至 `'test'`。
- **业务逻辑**: 载入对应 `split` 下的 `json/jsonl` 表单，使用 `augment` 和 `features` 包内的构建器初始化所有的增广流水和 `feature_extractors` 对象。

#### `_load_data_list(self, list_path: str) -> List[dict]`
- **输入**: 数据清单文件路径 `list_path`。
- **输出**: 解析完毕的元数据字典列表。其中单行字典保证包含以下字段（依赖外界数据预处理脚本 `casia_process.py` 输入格式）:
  ```python
  {
      "audio_path": "相对路径或绝对路径", 
      "label": int_id, 
      "start_time_ms": int, # [可选] 默认 0
      "end_time_ms": int    # [可选] 默认 None
  }
  ```

#### `__getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]`
单点样本拉取总控。
- **业务逻辑**: 触发指针对长音频的 OOM 级懒加载读取与重采样 -> 单声道融合 -> 波形增强 (`wave_transform`) -> 触发多模态特征并发提取 (`feature_extractors`) 及频谱级增强掩码 (`spec_transform`) -> 计算出最终产物有效长度。
- **输出**: 包含三元素元组：
  - `extracted_features` (`Dict[str, torch.Tensor]`): 该样本提取出的特征包裹字典。结构依据 YAML 决定。
    > 假设配置中有 `log_mel` 和 `raw_waveform`：
    > `{ "log_mel": Tensor[F, T], "raw_waveform": Tensor[T] }` （张量被去除了 batch 占位维度）
  - `label` (`torch.Tensor`): `long` 类型，1D target。
  - `seq_length` (`int`): 该序列实际通过特征器映射后的时间维度大小 `T`，为后续 Attention 或 RNN 补零提供长度依据。

---

## 核心工厂类文件: `collate.py`

由于长短不一的数据组装需要定制化处理维度对齐，将所有的合批逻辑（Padding、动态掩码以及切窗操作）抽离为独立服务组件 `build_collate_fn` 存于此文件。

#### `build_collate_fn(config: dict) -> Callable`
- **输入**: 载入内存全局验证过的 `Yaml` 字典描述符。
- **输出**: 挂载有策略控制执行环境闭包的 `collate_fn` 方法引用。

### 返回的闭包方法: `collate_fn(batch)`
这是 DataLoader 每隔批次执行的真正合并逻辑函数。
- **输入**: 一笔数据清单 (`batch`)。结构为 `List[Tuple[Dict, Tensor, Int]]`（对应着由 `__getitem__` 吐出的元组数组）。
- **业务逻辑**: 基于 `audio_processing` 块配置产生对应的维数转换规则。包含目前 3 种主流策略的分水岭设计:
  
  1. **策略 `"truncate_pad"`**:
     - 从 `config` 获取 `max_frames` 常量。
     - 大于截断，短于前向填充右零补齐。
     - 为二维特征增加通道 `[B, 1, Freq, Time]` 以兼容 `Conv2D` 二维卷积算子。
  
  2. **策略 `"dynamic_mask"`**:
     - 通过比对找出当前所有该 `Batch` 中输入序列的 Max 时间轴长度。由于是完全基于当前块的长短做局部动态 Padding（无常量约束）。
     - 发行全局二进制 `boolean Mask Matrix (Attention Mask)`。
     - 调换轴为 Transformer 模型标准长列流: `[Batch, Time, Feature_dim]`。
     
  3. **策略 `"sliding_window"`** *(专为系统侧长语音不规整测试集与模型投票服务)*:
     - 不做整体缩放，依靠预先计算传入的 `stride` 与跨度对超长片段在最后的时间轴层层平移，物理开片为更细粒度的窗口数组集合 `all_windows`。
     - 对最终拉直展宽的模型侧特征维度，同频统计产出 `window_counts`（表明当前样本被裂变出了 N 份 window 切片），以及被复制 N 份一一映射的张量新伪目标 `expanded_labels`。

- **输出 (依据策略异构返回以下字段集)**:
  - 正常返回示例（如遇到 `truncate_pad`）:
     ```python
     (
         {'x': {'log_mel': Tensor(B, 1, F, T)}},
         labels: Tensor(B,)
     )
     ```
  - 特殊推理投票返回（如开启 `sliding_window`）：
     ```python
     (
         {
             'x': {'log_mel': Tensor(Total_Windows, 1, F, T)}, 
             'window_counts': Tensor(B,),         # 每个文件原始对应的切片分配长度
             'original_labels': Tensor(B,)        # 原生长度用于计算投票通过率评比的标签集
         },
         labels: Tensor(Total_Windows,)  # 扩展开拉直的标签
     )
     ```

---

## 进阶组件库

### 1. `augment/` 数据增强算子包

包含在音频不同空间状态下的正则化与泛化防御算子。流水线的装配由 `augment/builder.py` 工厂对外统一暴露。

#### `augment/time_domain.py`
针对 1D `Waveform` 浮点数张量进行物理信号层面的扰乱操作。包含：
- `AddGaussianNoise(snr, p)`: 注入受控信噪比的高斯白噪，增强对低信噪比环境的鲁棒性。
- `PitchShift(sample_rate, n_steps, p)`: 进行音高偏移（支持 Torchaudio 内置特征，降级使用重采样），模拟说话人情感张力波动。
- `TimeStretch(rate, p)`: 使用相位声码器支持的时间拉伸（不变调），模拟不同语速。
- `TimeShift(shift_max_ratio, p)`: 时间轴上的伪平移截断对齐补零机制。
- `VolumeScale(gain_min, gain_max, p)`: 音量按随机倍率浮动缩放。
- *(预留)* `RIRSimulation` 和 `DynamicSNRMixing`: 高级环境混响与背景噪声源混合（课程高阶预留）。

#### `augment/freq_domain.py`
针对经过特征提取的 2D `Spectrogram` 进行谱域能量平面的扰动：
- `SpecMasking(time_mask_param, freq_mask_param, p)`: 原生 SpecAugment，时频二维涂抹抹零正则化。
- `FilterAugment(n_band, db_range, band_width_ratio, p)`: 频段截断掩盖的柔和拉伸版本。
- `VTLP(warp_factor_range, p)`: 声道长度扰动，针对频谱频率轴的特征缩放近似。
- *(预留)* `SpecMix(p)`: 批次级 (Batch-level) 拼图特征增强。

---

### 2. `features/` 多模态特征提取包

项目解耦的底层特征计算模块栈。通过 `features/builder.py` 内部工厂可灵活地实例化并在不同网络拓扑中任意组合。支持包括传统声学特征、时频能量、以及发音人微观病理学等超过 10 种特征计算过程。

#### `features/time_domain.py`
1D 本源时域/韵律及音质解析器：
- **韵律类**:
  - `PitchF0`: 基于 NCCF 的短时高精度基频抽取，直观表征情绪起伏与发音张力。
  - `RMS`: 短时能量均方根获取，代表信号的绝对声压或感知响度激荡。
  - `ZeroCrossingRate (ZCR)`: 信号过零速率评估，刻画清音、爆破音密度比例及含噪量。
- **音质类**:
  - `JitterShimmerHNR`: 联合计算依赖周期的声带微观抖动与谐波噪比 (仅基础实现)。

#### `features/freq_domain.py`
基于短时傅里叶变换及其衍生映射谱面的 2D 系特征算子：
- `MFCC`: 经典的声学倒谱系数表示通道内涵。
- `SpectralCentroid`: 频谱质心计算，量化情感频带的中心点转移规律。
- `SpectralRolloff`: 85%能量截断点频标提取。
- `SpectralFlatness`: 量化频带内的平坦几何与算数平均离散程度。
- `SpectralFlux`: 计算谱级一阶时间差分能量激变值，追踪起音与截音。
- `ComputeDeltas`: 附带挂件，可为任意基础二维特征矩阵增加一阶差分 (Delta) 与二阶差分 (Delta-Delta) 通道层级。
