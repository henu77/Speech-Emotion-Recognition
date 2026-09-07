"""RawWaveform 表示：输出原始波形 ``[T]``（设计文档 §9.2）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import (
    LAYOUT_T,
    AudioData,
    RepresentationOutput,
    TensorSpec,
)


class RawWaveformConfig(BaseModel):
    """RawWaveform 无参数；保留空模型用于 schema 生成与未知参数报错。"""

    model_config = ConfigDict(extra="forbid")


class RawWaveform(Representation):
    """原始波形表示。

    输入 ``AudioData.waveform [1, T]``，输出 ``inputs["waveform"] [T]``，
    layout ``T``，length ``{"waveform": T}``。
    """

    descriptor = ComponentDescriptor(
        id="waveform",
        display_name="原始波形",
        category="representation",
        description="输出 [T] 原始波形，适用于端到端波形模型（Wav2Vec2、RawNet 等）。",
        config_schema=RawWaveformConfig.model_json_schema(),
        output_specs={"waveform": TensorSpec(layout=LAYOUT_T)},
    )

    def __init__(self) -> None:
        super().__init__()
        RawWaveformConfig()

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {"waveform": TensorSpec(layout=LAYOUT_T)}

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        waveform = audio.waveform[0]  # [1, T] -> [T]
        return RepresentationOutput(
            inputs={"waveform": waveform},
            lengths={"waveform": int(waveform.shape[0])},
        )
