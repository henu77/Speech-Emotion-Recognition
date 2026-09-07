"""Transform 公共设施：RandomApply 概率包装器与流水线（设计文档 §8.2）。

组件不应各自重复实现 ``p`` 判断；随机源使用 torch 全局 RNG，
DataLoader 多进程下通过 worker seed 保证可复现。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ser_lib.data.errors import TransformError
from ser_lib.data.types import TensorSpec


class RandomApply(nn.Module):
    """以概率 ``probability`` 应用内部 transform，否则原样返回。

    - 概率必须落在 ``[0, 1]``；
    - ``p=0`` 永不执行、``p=1`` 必定执行；
    - 使用 torch 全局 RNG 抽样，固定 seed 下可复现。
    """

    def __init__(self, transform: nn.Module, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise TransformError(
                f"RandomApply 概率必须在 [0, 1]，实际: {probability}",
                component=type(transform).__name__,
                stage="transform_build",
            )
        self.transform = transform
        self.probability = float(probability)

    def forward(self, *args, **kwargs):
        if torch.rand(()) >= self.probability:
            return args[0] if len(args) == 1 else args
        return self.transform(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"p={self.probability}"


class WaveformTransformPipeline(nn.Module):
    """波形级 transform 流水线：``[C, T] -> [C, T]``。"""

    def __init__(self, transforms: Sequence[nn.Module] | None = None) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms or []))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            try:
                waveform = transform(waveform)
            except TransformError:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一转业务异常
                raise TransformError(
                    f"波形 transform '{type(transform).__name__}' 执行失败: {exc}",
                    component=type(transform).__name__,
                    stage="waveform_transform",
                ) from exc
        return waveform


class FeatureTransformPipeline(nn.Module):
    """特征级 transform 流水线：对每个输入 key 应用全部 transform。

    构建（而不是运行）阶段已完成 layout 兼容性校验；运行时对每个
    temporal key 依次应用。
    """

    def __init__(self, transforms: Sequence[nn.Module] | None = None) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms or []))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.transforms:
            return inputs
        outputs = dict(inputs)
        for key, tensor in inputs.items():
            for transform in self.transforms:
                try:
                    tensor = transform(tensor)
                except TransformError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise TransformError(
                        f"特征 transform '{type(transform).__name__}' 处理输入 "
                        f"'{key}' 失败: {exc}",
                        component=type(transform).__name__,
                        stage="feature_transform",
                    ) from exc
            outputs[key] = tensor
        return outputs


def validate_feature_transform_layouts(
    transform: nn.Module,
    specs: dict[str, TensorSpec],
) -> None:
    """构建期校验特征 transform 与输入 layout 兼容（验证优先于运行）。

    transform 需要声明 ``compatible_layouts: tuple[str, ...]``；
    对所有时序输入 key 逐一检查。
    """
    compatible = getattr(transform, "compatible_layouts", None)
    if compatible is None:
        return
    incompatible = [
        (key, spec.layout)
        for key, spec in specs.items()
        if spec.temporal and spec.layout not in compatible
    ]
    if incompatible:
        raise TransformError(
            f"特征 transform '{type(transform).__name__}' 不兼容输入 layout "
            f"{incompatible}，支持的 layout: {compatible}",
            component=type(transform).__name__,
            stage="transform_build",
        )
