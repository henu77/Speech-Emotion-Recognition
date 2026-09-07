"""版本化实验配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from ser_lib.core.config import StrictConfig, load_versioned_config
from ser_lib.data.config import DataConfig

if TYPE_CHECKING:
    from ser_lib.data.audio import AudioLoader
    from ser_lib.data.collate import SERCollator
    from ser_lib.data.pipeline import SamplePipeline
    from ser_lib.models.base import SERModel


class ModelConfig(StrictConfig):
    """注册表模型及其构造参数。"""

    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(StrictConfig):
    """表示无关的训练循环配置。"""

    epochs: int = Field(default=10, ge=1)
    device: str = "cpu"
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    amp: bool = False
    gradient_clip_norm: float | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    checkpoint_dir: Path | None = None
    # 兼容旧调用；使用 ExperimentConfig 时由 optimizer 节点覆盖。
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class ExperimentConfig(StrictConfig):
    """一次可复现实验的完整、可序列化配置快照。"""

    schema_version: int = 1
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    optimizer: dict[str, Any] = Field(
        default_factory=lambda: {"type": "adamw", "params": {}}
    )
    scheduler: dict[str, Any] | None = None
    output_dir: Path = Path("runs/default")

    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, value: dict[str, Any]) -> dict[str, Any]:
        from ser_lib.engine.optim import parse_optimizer_config

        parse_optimizer_config(value)
        return value

    @field_validator("scheduler")
    @classmethod
    def _validate_scheduler(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        from ser_lib.engine.optim import parse_scheduler_config

        parse_scheduler_config(value)
        return value


@dataclass(frozen=True, slots=True)
class ExperimentComponents:
    """通过完整预检后可直接交给训练代码的运行时组件。"""

    model: "SERModel"
    audio_loader: "AudioLoader"
    pipeline: "SamplePipeline"
    collator: "SERCollator"


def build_experiment_components(
    config: ExperimentConfig,
    *,
    train: bool = True,
) -> ExperimentComponents:
    """构建实验组件，并在读取训练数据前完成全部静态兼容性检查。"""
    from ser_lib.data.collate import build_collator
    from ser_lib.data.pipeline import build_components
    from ser_lib.data.validation import validate_compatibility
    from ser_lib.models.registry import model_registry

    model_params = model_registry.validate_config(config.model.type, config.model.params)
    model = model_registry.create(config.model.type, **model_params)
    audio_loader, pipeline = build_components(config.data, train=train)
    validate_compatibility(
        pipeline.output_specs,
        model.model_spec,
        config.data.batching,
        num_classes=config.data.num_classes,
        sample_rate=config.data.audio.target_sample_rate,
    )
    collator = build_collator(pipeline.output_specs, config.data.batching)
    return ExperimentComponents(model, audio_loader, pipeline, collator)


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """加载 schema v1 实验配置。相对输出路径基于配置文件目录。"""
    config = load_versioned_config(path, ExperimentConfig, supported_versions={1})
    source = Path(path).expanduser().resolve()
    updates: dict[str, Any] = {}
    if not config.output_dir.is_absolute():
        updates["output_dir"] = (source.parent / config.output_dir).resolve()
    if config.trainer.checkpoint_dir is not None \
            and not config.trainer.checkpoint_dir.is_absolute():
        updates["trainer"] = config.trainer.model_copy(update={
            "checkpoint_dir": (source.parent / config.trainer.checkpoint_dir).resolve()
        })
    if not config.data.manifest.is_absolute():
        updates["data"] = config.data.model_copy(update={
            "manifest": (source.parent / config.data.manifest).resolve()
        })
    return config.model_copy(update=updates)


__all__ = [
    "ModelConfig", "TrainerConfig", "ExperimentConfig", "ExperimentComponents",
    "load_experiment_config", "build_experiment_components",
]
