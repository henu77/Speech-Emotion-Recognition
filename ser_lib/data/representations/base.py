"""Representation 基类（设计文档 §9.1）。

Representation 决定输入语义：waveform、Log-Mel、MFCC、F0 等是表示，不是
Dataset 类型。Representation 不读取文件、不处理 label、不知道 split，本身
必须是确定性的（随机操作属于 transform）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from ser_lib.data.errors import RepresentationError
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioData, RepresentationOutput, TensorSpec


class Representation(nn.Module, ABC):
    """输入表示的抽象基类。

    子类必须：

    - 声明类属性 ``descriptor``（供调用方枚举）；
    - 实现 ``output_specs``，返回 ``{key: TensorSpec}``；
    - 实现 ``forward(audio)`` 并保证输出满足自己声明的 specs。
    """

    descriptor: ComponentDescriptor

    @property
    @abstractmethod
    def output_specs(self) -> dict[str, TensorSpec]:
        """声明输出 tensor 的形状契约。"""
        ...

    @abstractmethod
    def forward(self, audio: AudioData) -> RepresentationOutput:
        """把 AudioData 转换为标准表示输出。"""
        ...

    def _require_mono(self, audio: AudioData) -> None:
        """第一版表示要求单声道输入（``[1, T]``）。"""
        if audio.waveform.shape[0] != 1:
            raise RepresentationError(
                f"表示 '{type(self).__name__}' 要求单声道 [1, T] 输入，实际 "
                f"[{audio.waveform.shape[0]}, {audio.waveform.shape[1]}]；"
                f"请在 AudioLoader 中开启 mono 或添加声道选择组件",
                path=audio.source_path,
                component=self.descriptor.id,
                stage="representation",
            )

    def _require_sample_rate(self, audio: AudioData) -> None:
        """校验配置采样率与 Loader 输出一致，不得静默以不同采样率计算。"""
        expected = self._expected_sample_rate
        if expected is not None and audio.sample_rate != expected:
            raise RepresentationError(
                f"表示 '{self.descriptor.id}' 配置的 sample_rate={expected} 与 "
                f"AudioLoader 输出 sample_rate={audio.sample_rate} 不一致；"
                f"请统一 AudioLoader.target_sample_rate 与表示配置",
                path=audio.source_path,
                component=self.descriptor.id,
                stage="representation",
            )

    _expected_sample_rate: int | None = None
