"""SER 模型公共契约。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from ser_lib.data.types import SERBatch
from ser_lib.data.validation import ModelSpec

@dataclass(frozen=True, slots=True)
class ModelOutput:
    """所有 SER 模型的标准输出。

    ``logits`` 必须是 ``[B, C]``；``embeddings`` 如存在必须是 ``[B, D]``；
    ``loss`` 如存在必须是标量。输出允许包含非有限 logits，以便训练器在统一
    位置产生诊断，但形状错误必须在模型边界立即失败。
    """

    logits: torch.Tensor
    embeddings: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor) or self.logits.dim() != 2:
            shape = tuple(self.logits.shape) if isinstance(self.logits, torch.Tensor) else None
            raise ValueError(f"ModelOutput.logits 必须是 [B,C] tensor，实际: {shape}")
        if self.logits.shape[0] < 1 or self.logits.shape[1] < 2:
            raise ValueError(f"ModelOutput.logits 要求 B>=1、C>=2，实际 {tuple(self.logits.shape)}")
        if self.embeddings is not None:
            if not isinstance(self.embeddings, torch.Tensor) or self.embeddings.dim() != 2 \
                    or self.embeddings.shape[0] != self.logits.shape[0]:
                raise ValueError(
                    "ModelOutput.embeddings 必须是与 logits batch 对齐的 [B,D] tensor"
                )
        if self.loss is not None and (
            not isinstance(self.loss, torch.Tensor) or self.loss.numel() != 1
        ):
            raise ValueError("ModelOutput.loss 必须是标量 tensor")

class SERModel(nn.Module, ABC):
    """所有内置及第三方适配模型必须实现的最小稳定接口。"""

    @property
    @abstractmethod
    def model_spec(self) -> ModelSpec: ...

    @property
    @abstractmethod
    def model_config(self) -> dict[str, Any]:
        """返回可 JSON 序列化、可用于注册表重建模型的完整配置。"""
        ...

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        """返回模型参数量；可限制为需要梯度的参数。"""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    @abstractmethod
    def forward(self, batch: SERBatch) -> ModelOutput: ...
