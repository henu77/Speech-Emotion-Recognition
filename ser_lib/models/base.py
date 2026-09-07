"""SER 模型公共契约。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import nn
from ser_lib.data.types import SERBatch
from ser_lib.data.validation import ModelSpec

@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor | None = None
    loss: torch.Tensor | None = None

class SERModel(nn.Module, ABC):
    @property
    @abstractmethod
    def model_spec(self) -> ModelSpec: ...

    @abstractmethod
    def forward(self, batch: SERBatch) -> ModelOutput: ...
