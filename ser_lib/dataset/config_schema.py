"""
SER 数据集配置 Schema 定义模块

使用 Pydantic V2 定义完整的分层配置 Schema，实现：
- 类型强约束：所有字段明确类型，枚举类使用 Literal
- 内置校验逻辑：互斥、隔离、必填、范围校验
- 兼容三类数据集：WaveformDataset, SpectrogramDataset, FeatureDataset

配置隔离规则：
- WaveformDataset: 不得包含 spectrogram 或 features 节点
- SpectrogramDataset: 必须包含 spectrogram 节点，不得包含 features 节点
- FeatureDataset: 必须包含 features 节点，不得包含 spectrogram 节点
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# 枚举类型定义
# =====================================================================

class DatasetType(str, Enum):
    """数据集类型枚举"""
    WAVEFORM = "waveform"
    SPECTROGRAM = "spectrogram"
    FEATURE = "feature"


class CollateStrategy(str, Enum):
    """批处理策略枚举"""
    TRUNCATE_PAD = "truncate_pad"
    DYNAMIC_MASK = "dynamic_mask"
    SLIDING_WINDOW = "sliding_window"


class SpectrogramType(str, Enum):
    """谱图类型枚举"""
    SPECTROGRAM = "Spectrogram"
    MEL_SPECTROGRAM = "MelSpectrogram"
    LOG_MEL_SPECTROGRAM = "LogMelSpectrogram"
    MFCC = "MFCC"


class FeatureType(str, Enum):
    """特征类型枚举"""
    # 谱图类
    SPECTROGRAM = "Spectrogram"
    MEL_SPECTROGRAM = "MelSpectrogram"
    LOG_MEL_SPECTROGRAM = "LogMelSpectrogram"
    MFCC = "MFCC"
    # 时域韵律特征
    F0 = "F0"
    RMS = "RMS"
    ZCR = "ZCR"
    JITTER_SHIMMER_HNR = "JitterShimmerHNR"
    # 频域特征
    SPECTRAL_CENTROID = "SpectralCentroid"
    SPECTRAL_ROLLOFF = "SpectralRolloff"
    SPECTRAL_FLATNESS = "SpectralFlatness"
    SPECTRAL_FLUX = "SpectralFlux"
    DELTA = "Delta"


class WaveformAugmentType(str, Enum):
    """波形级增强类型枚举"""
    NORMALIZE = "Normalize"
    ADD_GAUSSIAN_NOISE = "AddGaussianNoise"
    PITCH_SHIFT = "PitchShift"
    TIME_STRETCH = "TimeStretch"
    TIME_SHIFT = "TimeShift"
    VOLUME_SCALE = "VolumeScale"
    RIR_SIMULATION = "RIR_Simulation"
    DYNAMIC_SNR_MIXING = "DynamicSNRMixing"


class SpectrogramAugmentType(str, Enum):
    """频域增强类型枚举"""
    SPEC_MASKING = "SpecMasking"
    FILTER_AUGMENT = "FilterAugment"
    VTLP = "VTLP"
    SPEC_MIX = "SpecMix"


class BatchAugmentType(str, Enum):
    """批次级增强类型枚举"""
    MIXUP = "Mixup"


# =====================================================================
# 基础配置 Schema
# =====================================================================

class ClassMappingItem(BaseModel):
    """单个类别映射项"""
    en: str = Field(..., description="英文标签")
    zh: str = Field(..., description="中文标签")


class PathConfig(BaseModel):
    """路径配置"""
    metadata_dir: Optional[str] = Field(None, description="元数据目录路径")
    data_root_dir: Optional[str] = Field(None, description="音频数据根目录")


class DataListsConfig(BaseModel):
    """数据列表配置"""
    train: str = Field(..., description="训练集 JSONL 文件名")
    val: str = Field(..., description="验证集 JSONL 文件名")
    test: str = Field(..., description="测试集 JSONL 文件名")


class AudioConfig(BaseModel):
    """音频基础参数配置"""
    target_sr: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="目标采样率 (Hz)"
    )


class CollateConfig(BaseModel):
    """批处理策略配置"""
    strategy: CollateStrategy = Field(
        default=CollateStrategy.TRUNCATE_PAD,
        description="批处理对齐策略"
    )
    max_frames: int = Field(
        default=300,
        ge=1,
        description="最大帧数 [truncate_pad 专用]"
    )
    window_size: int = Field(
        default=300,
        ge=1,
        description="滑动窗口大小 [sliding_window 专用]"
    )
    stride: int = Field(
        default=150,
        ge=1,
        description="滑动窗口步长 [sliding_window 专用]"
    )

    @model_validator(mode='after')
    def validate_stride(self) -> 'CollateConfig':
        """校验 stride <= window_size"""
        if self.stride > self.window_size:
            raise ValueError(
                f"[CollateConfig] stride ({self.stride}) 不能大于 window_size ({self.window_size})"
            )
        return self


# =====================================================================
# 特征配置 Schema
# =====================================================================

class SpectrogramKwargs(BaseModel):
    """谱图提取参数"""
    sample_rate: Optional[int] = Field(None, ge=8000, le=48000)
    n_fft: Optional[int] = Field(None, ge=64, le=8192)
    win_length: Optional[int] = Field(None, ge=64, le=8192)
    hop_length: Optional[int] = Field(None, ge=16, le=4096)
    n_mels: Optional[int] = Field(None, ge=20, le=256)
    f_min: Optional[float] = Field(None, ge=0.0)
    f_max: Optional[float] = Field(None, ge=1000.0, le=24000.0)
    power: Optional[float] = Field(None, ge=1.0, le=3.0)
    top_db: Optional[float] = Field(None, ge=40.0, le=120.0)
    n_mfcc: Optional[int] = Field(None, ge=10, le=100)

    @model_validator(mode='after')
    def validate_fft_params(self) -> 'SpectrogramKwargs':
        """校验 FFT 相关参数"""
        if self.win_length is not None and self.n_fft is not None:
            if self.win_length > self.n_fft:
                raise ValueError(
                    f"[SpectrogramKwargs] win_length ({self.win_length}) 不能大于 n_fft ({self.n_fft})"
                )
        if self.f_min is not None and self.f_max is not None:
            if self.f_min >= self.f_max:
                raise ValueError(
                    f"[SpectrogramKwargs] f_min ({self.f_min}) 必须小于 f_max ({self.f_max})"
                )
        return self


class SpectrogramFeatureConfig(BaseModel):
    """谱图特征配置 (SpectrogramDataset 专属)"""
    type: SpectrogramType = Field(..., description="谱图类型")
    kwargs: SpectrogramKwargs = Field(default_factory=SpectrogramKwargs, description="谱图参数")


class FeatureItemConfig(BaseModel):
    """单个特征配置项"""
    type: FeatureType = Field(..., description="特征类型")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="特征参数")


class MultiFeatureConfig(BaseModel):
    """多特征配置 (FeatureDataset 专属)"""
    selected_features: Dict[str, FeatureItemConfig] = Field(
        ...,
        min_length=1,
        description="特征字典，key 为特征名，value 为特征配置"
    )


# =====================================================================
# 数据增强配置 Schema
# =====================================================================

class WaveformAugmentItem(BaseModel):
    """波形级增强项"""
    type: WaveformAugmentType = Field(..., description="增强类型")
    p: float = Field(default=0.5, ge=0.0, le=1.0, description="触发概率")
    # 各增强类型的参数
    snr: Optional[float] = Field(None, description="信噪比 (dB) [AddGaussianNoise]")
    n_steps: Optional[int] = Field(None, description="半音阶数 [PitchShift]")
    rate: Optional[float] = Field(None, gt=0.0, description="速率 [TimeStretch]")
    shift_max_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="最大平移比例 [TimeShift]")
    gain_min: Optional[float] = Field(None, description="最小增益 [VolumeScale]")
    gain_max: Optional[float] = Field(None, description="最大增益 [VolumeScale]")
    rir_path: Optional[str] = Field(None, description="RIR 文件路径 [RIR_Simulation]")
    noise_dataset_path: Optional[str] = Field(None, description="噪声数据集路径 [DynamicSNRMixing]")

    @model_validator(mode='after')
    def validate_params(self) -> 'WaveformAugmentItem':
        """校验增强参数"""
        if self.type == WaveformAugmentType.ADD_GAUSSIAN_NOISE:
            if self.snr is None:
                raise ValueError(f"[WaveformAugmentItem] AddGaussianNoise 需要指定 snr 参数")
        if self.type == WaveformAugmentType.PITCH_SHIFT:
            if self.n_steps is None:
                raise ValueError(f"[WaveformAugmentItem] PitchShift 需要指定 n_steps 参数")
        if self.type == WaveformAugmentType.TIME_STRETCH:
            if self.rate is None:
                raise ValueError(f"[WaveformAugmentItem] TimeStretch 需要指定 rate 参数")
        if self.type == WaveformAugmentType.TIME_SHIFT:
            if self.shift_max_ratio is None:
                raise ValueError(f"[WaveformAugmentItem] TimeShift 需要指定 shift_max_ratio 参数")
        if self.type == WaveformAugmentType.VOLUME_SCALE:
            if self.gain_min is not None and self.gain_max is not None:
                if self.gain_min >= self.gain_max:
                    raise ValueError(
                        f"[WaveformAugmentItem] gain_min ({self.gain_min}) 必须小于 gain_max ({self.gain_max})"
                    )
        return self


class SpectrogramAugmentItem(BaseModel):
    """频域增强项"""
    type: SpectrogramAugmentType = Field(..., description="增强类型")
    p: float = Field(default=0.5, ge=0.0, le=1.0, description="触发概率")
    time_mask_param: Optional[int] = Field(None, ge=1, description="时间掩码宽度 [SpecMasking]")
    freq_mask_param: Optional[int] = Field(None, ge=1, description="频率掩码宽度 [SpecMasking]")
    n_band: Optional[int] = Field(None, ge=1, description="频带数量 [FilterAugment]")
    db_range: Optional[List[float]] = Field(None, description="增益范围 [FilterAugment]")
    band_width_ratio: Optional[float] = Field(None, gt=0.0, le=1.0, description="频带宽度比例 [FilterAugment]")
    warp_factor_range: Optional[List[float]] = Field(None, description="扭曲因子范围 [VTLP]")

    @model_validator(mode='after')
    def validate_params(self) -> 'SpectrogramAugmentItem':
        """校验增强参数"""
        if self.type == SpectrogramAugmentType.SPEC_MASKING:
            if self.time_mask_param is None or self.freq_mask_param is None:
                raise ValueError(f"[SpectrogramAugmentItem] SpecMasking 需要指定 time_mask_param 和 freq_mask_param")
        if self.type == SpectrogramAugmentType.FILTER_AUGMENT:
            if self.db_range is not None and len(self.db_range) != 2:
                raise ValueError(f"[SpectrogramAugmentItem] db_range 必须是长度为 2 的列表")
        if self.type == SpectrogramAugmentType.VTLP:
            if self.warp_factor_range is not None and len(self.warp_factor_range) != 2:
                raise ValueError(f"[SpectrogramAugmentItem] warp_factor_range 必须是长度为 2 的列表")
        return self


class BatchAugmentItem(BaseModel):
    """批次级增强项"""
    type: BatchAugmentType = Field(..., description="增强类型")
    p: float = Field(default=0.5, ge=0.0, le=1.0, description="触发概率")
    alpha: Optional[float] = Field(None, gt=0.0, description="Beta 分布参数 [Mixup]")


class SplitAugmentConfig(BaseModel):
    """单个数据划分的增强配置"""
    train: List[Union[WaveformAugmentItem, SpectrogramAugmentItem, BatchAugmentItem]] = Field(
        default_factory=list,
        description="训练集增强列表"
    )
    val: List[Union[WaveformAugmentItem, SpectrogramAugmentItem, BatchAugmentItem]] = Field(
        default_factory=list,
        description="验证集增强列表"
    )
    test: List[Union[WaveformAugmentItem, SpectrogramAugmentItem, BatchAugmentItem]] = Field(
        default_factory=list,
        description="测试集增强列表"
    )


class WaveformAugmentConfig(BaseModel):
    """波形级增强配置"""
    train: Optional[List[WaveformAugmentItem]] = Field(default_factory=list)
    val: Optional[List[WaveformAugmentItem]] = Field(default_factory=list)
    test: Optional[List[WaveformAugmentItem]] = Field(default_factory=list)

    @field_validator('train', 'val', 'test', mode='before')
    @classmethod
    def convert_none_to_list(cls, v):
        """将 None 转换为空列表"""
        return v if v is not None else []


class SpectrogramAugmentConfig(BaseModel):
    """频域增强配置"""
    train: Optional[List[SpectrogramAugmentItem]] = Field(default_factory=list)
    val: Optional[List[SpectrogramAugmentItem]] = Field(default_factory=list)
    test: Optional[List[SpectrogramAugmentItem]] = Field(default_factory=list)

    @field_validator('train', 'val', 'test', mode='before')
    @classmethod
    def convert_none_to_list(cls, v):
        """将 None 转换为空列表"""
        return v if v is not None else []


class BatchAugmentConfig(BaseModel):
    """批次级增强配置"""
    train: Optional[List[BatchAugmentItem]] = Field(default_factory=list)
    val: Optional[List[BatchAugmentItem]] = Field(default_factory=list)
    test: Optional[List[BatchAugmentItem]] = Field(default_factory=list)

    @field_validator('train', 'val', 'test', mode='before')
    @classmethod
    def convert_none_to_list(cls, v):
        """将 None 转换为空列表"""
        return v if v is not None else []


class TransformsConfig(BaseModel):
    """完整数据增强配置"""
    waveform_level: WaveformAugmentConfig = Field(
        default_factory=WaveformAugmentConfig,
        description="波形级基础增强"
    )
    advanced_waveform_level: WaveformAugmentConfig = Field(
        default_factory=WaveformAugmentConfig,
        description="波形级高级增强"
    )
    spectrogram_level: SpectrogramAugmentConfig = Field(
        default_factory=SpectrogramAugmentConfig,
        description="频域增强"
    )
    batch_level: BatchAugmentConfig = Field(
        default_factory=BatchAugmentConfig,
        description="批次级增强"
    )


# =====================================================================
# 全局配置 Schema (带互斥校验)
# =====================================================================

class BaseDatasetConfig(BaseModel):
    """数据集配置基类"""

    # 必填字段
    dataset_name: str = Field(..., min_length=1, description="数据集名称")
    num_classes: int = Field(..., ge=2, le=100, description="类别数量")
    class_mapping: Dict[int, ClassMappingItem] = Field(..., description="类别映射字典")
    data_lists: DataListsConfig = Field(..., description="数据列表配置")

    # 可选字段
    paths: Optional[PathConfig] = Field(None, description="路径配置")
    audio: AudioConfig = Field(default_factory=AudioConfig, description="音频参数")
    audio_processing: CollateConfig = Field(default_factory=CollateConfig, description="批处理策略")
    transforms: TransformsConfig = Field(default_factory=TransformsConfig, description="数据增强配置")

    # 特征配置 (互斥)
    spectrogram: Optional[SpectrogramFeatureConfig] = Field(None, description="谱图特征配置 [SpectrogramDataset 专属]")
    features: Optional[MultiFeatureConfig] = Field(None, description="多特征配置 [FeatureDataset 专属]")

    @field_validator('class_mapping')
    @classmethod
    def validate_class_mapping(cls, v: Dict[int, ClassMappingItem]) -> Dict[int, ClassMappingItem]:
        """校验 class_mapping 的键从 0 开始连续"""
        keys = sorted(v.keys())
        expected = list(range(len(keys)))
        if keys != expected:
            raise ValueError(
                f"[class_mapping] 类别 ID 必须从 0 开始连续，期望 {expected}，实际 {keys}"
            )
        return v

    @field_validator('num_classes')
    @classmethod
    def validate_num_classes(cls, v: int, info) -> int:
        """校验 num_classes 与 class_mapping 一致"""
        if 'class_mapping' in info.data:
            mapping_len = len(info.data['class_mapping'])
            if v != mapping_len:
                raise ValueError(
                    f"[num_classes] num_classes ({v}) 必须与 class_mapping 数量 ({mapping_len}) 一致"
                )
        return v

    @model_validator(mode='after')
    def validate_feature_mutex(self) -> 'BaseDatasetConfig':
        """
        校验特征配置互斥规则：
        - spectrogram 与 features 不得同时存在
        """
        if self.spectrogram is not None and self.features is not None:
            raise ValueError(
                "[配置冲突] spectrogram 与 features 节点互斥，同一配置文件中仅能存在其中一个"
            )
        return self


class WaveformDatasetConfig(BaseDatasetConfig):
    """
    WaveformDataset 配置 Schema

    【配置隔离规则】
    - 不得包含 spectrogram 节点
    - 不得包含 features 节点
    """

    @model_validator(mode='after')
    def validate_waveform_isolation(self) -> 'WaveformDatasetConfig':
        """校验 WaveformDataset 配置隔离规则"""
        if self.spectrogram is not None:
            raise ValueError(
                "[配置隔离] WaveformDataset 配置文件不得包含 spectrogram 节点"
            )
        if self.features is not None:
            raise ValueError(
                "[配置隔离] WaveformDataset 配置文件不得包含 features 节点"
            )
        return self


class SpectrogramDatasetConfig(BaseDatasetConfig):
    """
    SpectrogramDataset 配置 Schema

    【配置隔离规则】
    - 必须包含 spectrogram 节点
    - 不得包含 features 节点
    """

    @model_validator(mode='after')
    def validate_spectrogram_isolation(self) -> 'SpectrogramDatasetConfig':
        """校验 SpectrogramDataset 配置隔离规则"""
        if self.spectrogram is None:
            raise ValueError(
                "[配置隔离] SpectrogramDataset 配置文件必须包含 spectrogram 节点"
            )
        if self.features is not None:
            raise ValueError(
                "[配置隔离] SpectrogramDataset 配置文件不得包含 features 节点"
            )
        return self


class FeatureDatasetConfig(BaseDatasetConfig):
    """
    FeatureDataset 配置 Schema

    【配置隔离规则】
    - 必须包含 features 节点
    - 不得包含 spectrogram 节点
    """

    @model_validator(mode='after')
    def validate_feature_isolation(self) -> 'FeatureDatasetConfig':
        """校验 FeatureDataset 配置隔离规则"""
        if self.features is None:
            raise ValueError(
                "[配置隔离] FeatureDataset 配置文件必须包含 features 节点"
            )
        if self.spectrogram is not None:
            raise ValueError(
                "[配置隔离] FeatureDataset 配置文件不得包含 spectrogram 节点"
            )
        return self


# =====================================================================
# 配置加载工具函数
# =====================================================================

def detect_dataset_type(config_dict: Dict[str, Any]) -> DatasetType:
    """
    根据配置字典自动检测数据集类型

    Args:
        config_dict: 原始配置字典

    Returns:
        DatasetType: 数据集类型枚举值

    Raises:
        ValueError: 无法确定数据集类型时抛出
    """
    has_spectrogram = 'spectrogram' in config_dict and config_dict['spectrogram'] is not None
    has_features = 'features' in config_dict and config_dict['features'] is not None

    if has_spectrogram and has_features:
        raise ValueError(
            "[配置冲突] spectrogram 与 features 节点互斥，无法确定数据集类型"
        )

    if has_spectrogram:
        return DatasetType.SPECTROGRAM
    elif has_features:
        return DatasetType.FEATURE
    else:
        return DatasetType.WAVEFORM


def load_config(config_path: str) -> BaseDatasetConfig:
    """
    加载并校验配置文件

    自动检测数据集类型并返回对应的配置 Schema 实例。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        BaseDatasetConfig: 配置 Schema 实例 (实际类型为 WaveformDatasetConfig /
                           SpectrogramDatasetConfig / FeatureDatasetConfig)

    Raises:
        ValidationError: 配置校验失败时抛出
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    dataset_type = detect_dataset_type(config_dict)

    if dataset_type == DatasetType.WAVEFORM:
        return WaveformDatasetConfig(**config_dict)
    elif dataset_type == DatasetType.SPECTROGRAM:
        return SpectrogramDatasetConfig(**config_dict)
    else:
        return FeatureDatasetConfig(**config_dict)


def validate_config_dict(config_dict: Dict[str, Any]) -> BaseDatasetConfig:
    """
    校验配置字典

    Args:
        config_dict: 原始配置字典

    Returns:
        BaseDatasetConfig: 配置 Schema 实例

    Raises:
        ValidationError: 配置校验失败时抛出
    """
    dataset_type = detect_dataset_type(config_dict)

    if dataset_type == DatasetType.WAVEFORM:
        return WaveformDatasetConfig(**config_dict)
    elif dataset_type == DatasetType.SPECTROGRAM:
        return SpectrogramDatasetConfig(**config_dict)
    else:
        return FeatureDatasetConfig(**config_dict)


# =====================================================================
# 导出
# =====================================================================

__all__ = [
    # 枚举
    'DatasetType',
    'CollateStrategy',
    'SpectrogramType',
    'FeatureType',
    'WaveformAugmentType',
    'SpectrogramAugmentType',
    'BatchAugmentType',
    # 配置 Schema
    'ClassMappingItem',
    'PathConfig',
    'DataListsConfig',
    'AudioConfig',
    'CollateConfig',
    'SpectrogramKwargs',
    'SpectrogramFeatureConfig',
    'FeatureItemConfig',
    'MultiFeatureConfig',
    'WaveformAugmentItem',
    'SpectrogramAugmentItem',
    'BatchAugmentItem',
    'WaveformAugmentConfig',
    'SpectrogramAugmentConfig',
    'BatchAugmentConfig',
    'TransformsConfig',
    'BaseDatasetConfig',
    'WaveformDatasetConfig',
    'SpectrogramDatasetConfig',
    'FeatureDatasetConfig',
    # 工具函数
    'detect_dataset_type',
    'load_config',
    'validate_config_dict',
]
