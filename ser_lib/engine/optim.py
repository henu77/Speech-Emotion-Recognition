"""白名单优化器和学习率调度器配置。"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field, model_validator

from ser_lib.core.config import StrictConfig


class AdamWConfig(StrictConfig):
    type: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.999, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)


class AdamConfig(StrictConfig):
    type: Literal["adam"] = "adam"
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.999, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)


class SGDConfig(StrictConfig):
    type: Literal["sgd"] = "sgd"
    learning_rate: float = Field(default=1e-2, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.0, ge=0, lt=1)
    nesterov: bool = False

    @model_validator(mode="after")
    def _validate_nesterov(self) -> "SGDConfig":
        if self.nesterov and self.momentum <= 0:
            raise ValueError("SGD nesterov=True 时 momentum 必须 > 0")
        return self


OptimizerConfig = AdamWConfig | AdamConfig | SGDConfig


class StepSchedulerConfig(StrictConfig):
    type: Literal["step"] = "step"
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0, le=1)


class CosineSchedulerConfig(StrictConfig):
    type: Literal["cosine"] = "cosine"
    t_max: int = Field(ge=1)
    eta_min: float = Field(default=0.0, ge=0)


SchedulerConfig = StepSchedulerConfig | CosineSchedulerConfig


def parse_optimizer_config(raw: dict[str, Any]) -> OptimizerConfig:
    """解析白名单优化器，未知类型和参数立即失败。"""
    kind = raw.get("type", "adamw")
    params = raw.get("params", {})
    if set(raw) - {"type", "params"}:
        raise ValueError(f"optimizer 包含未知字段: {sorted(set(raw) - {'type', 'params'})}")
    if not isinstance(params, dict):
        raise ValueError("optimizer.params 必须是映射")
    models = {"adamw": AdamWConfig, "adam": AdamConfig, "sgd": SGDConfig}
    if kind not in models:
        raise ValueError(f"未知 optimizer.type={kind!r}，可用: {sorted(models)}")
    return models[kind](type=kind, **params)


def build_optimizer(
    parameters,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    common = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if isinstance(config, AdamWConfig):
        return torch.optim.AdamW(
            parameters, **common, betas=(config.beta1, config.beta2), eps=config.eps
        )
    if isinstance(config, AdamConfig):
        return torch.optim.Adam(
            parameters, **common, betas=(config.beta1, config.beta2), eps=config.eps
        )
    if isinstance(config, SGDConfig):
        return torch.optim.SGD(
            parameters, **common, momentum=config.momentum, nesterov=config.nesterov
        )
    raise TypeError(f"不支持的优化器配置: {type(config)!r}")


def parse_scheduler_config(raw: dict[str, Any] | None) -> SchedulerConfig | None:
    if raw is None:
        return None
    kind = raw.get("type")
    params = raw.get("params", {})
    if set(raw) - {"type", "params"}:
        raise ValueError(f"scheduler 包含未知字段: {sorted(set(raw) - {'type', 'params'})}")
    if not isinstance(params, dict):
        raise ValueError("scheduler.params 必须是映射")
    models = {"step": StepSchedulerConfig, "cosine": CosineSchedulerConfig}
    if kind not in models:
        raise ValueError(f"未知 scheduler.type={kind!r}，可用: {sorted(models)}")
    return models[kind](type=kind, **params)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config is None:
        return None
    if isinstance(config, StepSchedulerConfig):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    if isinstance(config, CosineSchedulerConfig):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max, eta_min=config.eta_min
        )
    raise TypeError(f"不支持的调度器配置: {type(config)!r}")


__all__ = [
    "AdamWConfig", "AdamConfig", "SGDConfig", "OptimizerConfig",
    "StepSchedulerConfig", "CosineSchedulerConfig", "SchedulerConfig",
    "parse_optimizer_config", "build_optimizer", "parse_scheduler_config", "build_scheduler",
]
