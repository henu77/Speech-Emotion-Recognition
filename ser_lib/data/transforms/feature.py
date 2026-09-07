"""特征级 transform（设计文档 §8.1、T3.3）。

stable：spec_masking（SpecAugment 时间/频率掩码）。
兼容 layout：``FT``（``[F, T]``）与 ``CFT``（``[C, F, T]``）——torchaudio
masking 约定最后一维为时间、倒数第二维为频率。
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from pydantic import BaseModel, ConfigDict, Field

from ser_lib.data.registry import ComponentDescriptor


class SpecMaskingConfig(BaseModel):
    """SpecAugment 掩码参数。"""

    model_config = ConfigDict(extra="forbid")

    time_mask_param: int = Field(default=30, ge=1, description="时间掩码最大宽度")
    freq_mask_param: int = Field(default=15, ge=1, description="频率掩码最大宽度")


class SpecMasking(torch.nn.Module):
    """时间掩码 + 频率掩码（分别评估触发概率由 RandomApply 包装器决定）。"""

    compatible_layouts = ("FT", "CFT")

    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 15) -> None:
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.time_masking(features)
        features = self.freq_masking(features)
        return features


SPEC_MASKING_DESCRIPTOR = ComponentDescriptor(
    id="spec_masking",
    display_name="SpecAugment 掩码",
    category="feature_transform",
    description="对 [F, T] 或 [C, F, T] 输入应用时间与频率掩码。",
    config_schema=SpecMaskingConfig.model_json_schema(),
)

__all__ = ["SpecMasking", "SpecMaskingConfig", "SPEC_MASKING_DESCRIPTOR"]
