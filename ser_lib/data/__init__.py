"""ser_lib.data：数据加载与表示系统（新核心）。

设计文档：``docs/DATA_PIPELINE_REFACTOR_PLAN.md``。

四个核心边界：

- **数据集只描述样本集合**（:class:`SERDataset` 只组装样本）；
- **Representation 描述输入形式**（waveform、Log-Mel、MFCC 是表示，
  不是 Dataset 类型）；
- **TensorSpec 描述形状契约**（key、layout、length、mask 均有明确规格）；
- **Collator 根据规格批处理**（不根据 Dataset 类型分支）。

import 本包即完成组件注册（注册表操作轻量，不扫描文件、不初始化 CUDA）。
"""

from ser_lib.data.audio import AudioLoader, AudioLoaderConfig
from ser_lib.data.collate import CollateStrategy, SERCollator, build_collator
from ser_lib.data.cache import CachedRepresentation
from ser_lib.data.config import (
    AudioSettings,
    BatchingConfig,
    CacheSettings,
    ComponentConfig,
    DataConfig,
    load_data_config,
)
from ser_lib.data.dataset import SERDataset
from ser_lib.data.errors import (
    AudioDecodeError,
    AudioNotFoundError,
    CollationError,
    CompatibilityError,
    InvalidAudioSegmentError,
    ManifestError,
    RegistryError,
    RepresentationError,
    SERDataError,
    TransformError,
)
from ser_lib.data.importers import (
    CasiaImporter,
    CsvImporter,
    FolderImporter,
    ImportIssue,
    ImportPreview,
    JsonlImporter,
    RavdessImporter,
    register_importers,
)
from ser_lib.data.manifest import (
    DatasetManifest,
    ManifestMeta,
    read_jsonl,
    write_jsonl,
)
from ser_lib.data.pipeline import SamplePipeline, build_components, build_pipeline
from ser_lib.data.profiling import (
    AudioProbeFailure,
    DatasetAudioProfile,
    profile_manifest_audio,
)
from ser_lib.data.registry import (
    ComponentDescriptor,
    Registry,
    default_registry,
)
from ser_lib.data.representations import register_representations
from ser_lib.data.transforms import register_transforms
from ser_lib.data.types import (
    AudioData,
    AudioRecord,
    RepresentationOutput,
    SERBatch,
    SERSample,
    TensorSpec,
    validate_sample_contract,
)
from ser_lib.data.validation import ModelSpec, validate_compatibility

# 注册全部内置组件（import 即可用，代价为常数时间字典操作）
register_importers()
register_representations()
register_transforms()

__all__ = [
    # 核心类型
    "AudioRecord",
    "AudioData",
    "TensorSpec",
    "RepresentationOutput",
    "SERSample",
    "SERBatch",
    "validate_sample_contract",
    # 异常
    "SERDataError",
    "ManifestError",
    "AudioNotFoundError",
    "AudioDecodeError",
    "InvalidAudioSegmentError",
    "RepresentationError",
    "TransformError",
    "CollationError",
    "CompatibilityError",
    "RegistryError",
    # Manifest
    "DatasetManifest",
    "ManifestMeta",
    "read_jsonl",
    "write_jsonl",
    # Audio
    "AudioLoader",
    "AudioLoaderConfig",
    # Importers
    "FolderImporter",
    "CsvImporter",
    "JsonlImporter",
    "CasiaImporter",
    "ImportPreview",
    "ImportIssue",
    "RavdessImporter",
    "register_importers",
    # Pipeline / Dataset
    "SamplePipeline",
    "SERDataset",
    "build_pipeline",
    "build_components",
    # Collate
    "SERCollator",
    "CollateStrategy",
    "build_collator",
    "CachedRepresentation",
    # Registry
    "Registry",
    "default_registry",
    "ComponentDescriptor",
    "register_representations",
    "register_transforms",
    # Config
    "DataConfig",
    "ComponentConfig",
    "AudioSettings",
    "CacheSettings",
    "BatchingConfig",
    "load_data_config",
    # Validation
    "ModelSpec",
    "validate_compatibility",
    "AudioProbeFailure", "DatasetAudioProfile", "profile_manifest_audio",
]
