"""数据模块强类型配置（设计文档 §12.1）。

顶层配置负责结构校验；每个组件自己的参数 Schema 由注册表中的 Pydantic
配置模型负责，不在中央配置中枚举所有未来参数。所有配置模型默认
``extra="forbid"``，防止拼错参数被静默忽略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 与 ser_lib/data/collate.py 中的 CollateStrategy 枚举保持一致
BatchingType = Literal["dynamic", "fixed", "sliding"]


class _StrictModel(BaseModel):
    """全库配置模型基类：禁止未知字段。"""

    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_StrictModel):
    """通用组件引用：``{type, params, probability}``。

    ``probability`` 仅用于随机 transform（RandomApply 包装器，§8.2）；
    其余组件必须留空。
    """

    type: str = Field(..., min_length=1, description="注册表中的组件名")
    params: dict[str, Any] = Field(default_factory=dict, description="组件参数")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="随机 transform 的触发概率"
    )


class AudioSettings(_StrictModel):
    """AudioLoader 的可序列化配置（对应运行时 AudioLoaderConfig）。"""

    target_sample_rate: int = Field(default=16000, ge=1000, le=192000)
    mono: bool = True
    normalize_peak: bool = False
    backend: Literal["torchaudio"] = "torchaudio"


class CacheSettings(_StrictModel):
    """确定性 Representation 磁盘缓存。"""

    enabled: bool = False
    directory: Path = Path(".ser-cache/features")


class FixedBatching(_StrictModel):
    """固定长度批处理参数：按 key 配置最大长度。"""

    max_lengths: dict[str, int] = Field(..., min_length=1)

    @field_validator("max_lengths")
    @classmethod
    def _positive(cls, v: dict[str, int]) -> dict[str, int]:
        for key, length in v.items():
            if length < 1:
                raise ValueError(f"max_lengths['{key}'] 必须 >= 1，实际: {length}")
        return v


class SlidingBatching(_StrictModel):
    """滑动窗口批处理参数。"""

    window_size: int = Field(..., ge=1)
    stride: int = Field(..., ge=1)

    @field_validator("stride")
    @classmethod
    def _stride_le_window(cls, v: int, info) -> int:
        window = info.data.get("window_size")
        if window is not None and v > window:
            raise ValueError(f"stride ({v}) 不能大于 window_size ({window})")
        return v


class BatchingConfig(_StrictModel):
    """批处理配置（设计文档 §11）。

    - ``dynamic``: 按 batch 内最大长度动态 padding 并生成 mask；
    - ``fixed``: 按 ``max_lengths``（每个 key 独立）截断/padding；
    - ``sliding``: 滑动窗口，仅支持一个主时序输入。
    """

    type: BatchingType = "dynamic"
    fixed: FixedBatching | None = None
    sliding: SlidingBatching | None = None
    # sliding 模式的主时序输入 key；多时序 key 时必须显式指定
    primary_key: str | None = None

    @field_validator("fixed", "sliding", mode="before")
    @classmethod
    def _none_to_missing(cls, v: Any) -> Any:
        return v

    @property
    def is_dynamic(self) -> bool:
        return self.type == "dynamic"

    def validate_completeness(self) -> None:
        """校验策略与参数节点的一致性，在任务启动前失败。"""
        if self.type == "fixed" and self.fixed is None:
            raise ValueError(
                "batching.type='fixed' 时必须提供 fixed.max_lengths "
                "（例如 {features: 300}）"
            )
        if self.type == "sliding" and self.sliding is None:
            raise ValueError(
                "batching.type='sliding' 时必须提供 sliding.window_size 与 sliding.stride"
            )
        if self.type != "fixed" and self.fixed is not None:
            raise ValueError("仅 batching.type='fixed' 允许提供 fixed 节点")
        if self.type != "sliding" and self.sliding is not None:
            raise ValueError("仅 batching.type='sliding' 允许提供 sliding 节点")


class DataConfig(_StrictModel):
    """数据模块顶层配置（设计文档 §15）。

    ``manifest`` 指向标准 ``dataset.yaml``；相对路径由 Manifest 层相对于
    配置文件所在目录解析，不依赖进程 cwd。
    """

    schema_version: int = 1
    manifest: Path
    dataset_id: str | None = None
    # 可选的类别表 {label_id: {en, zh, ...}}，用于兼容性校验
    labels: dict[int, dict[str, Any]] | None = None
    audio: AudioSettings = Field(default_factory=AudioSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    representation: ComponentConfig
    waveform_transforms: list[ComponentConfig] = Field(default_factory=list)
    feature_transforms: list[ComponentConfig] = Field(default_factory=list)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)

    @field_validator("batching")
    @classmethod
    def _validate_batching(cls, v: BatchingConfig) -> BatchingConfig:
        v.validate_completeness()
        return v

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: dict[int, dict[str, Any]] | None) -> dict[int, dict[str, Any]] | None:
        if v is None:
            return None
        keys = sorted(v.keys())
        if keys != list(range(len(keys))):
            raise ValueError(f"labels 的 key 必须从 0 开始连续，实际: {keys}")
        return v

    @property
    def num_classes(self) -> int | None:
        return len(self.labels) if self.labels is not None else None


def load_data_config(path: Path | str) -> DataConfig:
    """从 YAML 文件加载 DataConfig。

    路径相对语义：``manifest`` 字段相对于本配置文件所在目录解析。
    """
    import yaml

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {path}")
    raw = dict(raw)
    if "manifest" in raw and raw["manifest"] is not None:
        manifest_path = Path(str(raw["manifest"]))
        if not manifest_path.is_absolute():
            raw["manifest"] = (path.parent / manifest_path).resolve()
    return DataConfig(**raw)
