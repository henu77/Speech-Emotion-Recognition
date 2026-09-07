"""SER 数据模块核心公开类型（设计文档 §5，契约冻结）。

本模块定义四个边界中最基础的两个：

- ``TensorSpec`` 描述形状契约；
- ``AudioRecord`` / ``AudioData`` / ``RepresentationOutput`` / ``SERSample`` /
  ``SERBatch`` 描述单样本与批次的数据类型。

所有字段与验证规则如需修改，必须先同步更新设计文档与测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from ser_lib.data.errors import RepresentationError

# =====================================================================
# Layout 白名单（设计文档 §5.3）
# =====================================================================

LAYOUT_T = "T"      # 原始波形       单样本 [T]       批次 [B, T]
LAYOUT_FT = "FT"    # 频率/特征优先  单样本 [F, T]    批次 [B, F, T]
LAYOUT_TD = "TD"    # 时间优先       单样本 [T, D]    批次 [B, T, D]
LAYOUT_D = "D"      # 全局向量       单样本 [D]       批次 [B, D]
LAYOUT_CFT = "CFT"  # 多通道谱图     单样本 [C, F, T] 批次 [B, C, F, T]

ALLOWED_LAYOUTS: tuple[str, ...] = (LAYOUT_T, LAYOUT_FT, LAYOUT_TD, LAYOUT_D, LAYOUT_CFT)

# layout -> (单样本维度数, 时间轴索引 or None)
_LAYOUT_SHAPE_TABLE: dict[str, tuple[int, int | None]] = {
    LAYOUT_T: (1, 0),
    LAYOUT_FT: (2, 1),
    LAYOUT_TD: (2, 0),
    LAYOUT_D: (1, None),
    LAYOUT_CFT: (3, 2),
}


def time_axis_of(layout: str) -> int | None:
    """返回 layout 的时间轴索引；非时序 layout 返回 ``None``。"""
    _validate_layout_name(layout)
    return _LAYOUT_SHAPE_TABLE[layout][1]


def is_temporal(layout: str) -> bool:
    """layout 是否带时间轴。"""
    return time_axis_of(layout) is not None


def _validate_layout_name(layout: str) -> None:
    if layout not in ALLOWED_LAYOUTS:
        raise ValueError(
            f"未知 layout: {layout!r}。第一版仅支持白名单 {ALLOWED_LAYOUTS}，"
            f"不提供通用 layout 解析器。"
        )


# =====================================================================
# 数据类型
# =====================================================================


@dataclass(frozen=True, slots=True)
class AudioRecord:
    """Manifest 中的一条音频记录。

    Attributes:
        uid: 记录唯一 ID，在一个 manifest 中必须唯一。
        audio_path: 音频路径；可以是相对路径，解析由 Manifest/Resolver 完成。
        label: 类别标签；``None`` 表示无标签推理数据。
        start_ms: 片段起始毫秒（含），必须 ``>= 0``。
        end_ms: 片段结束毫秒（不含），与 ``start_ms`` 同时存在时必须 ``end_ms > start_ms``。
        speaker_id: 说话人 ID。
        sample_rate_hint: 数据源声明的原始采样率提示，可省去 probing。
        metadata: 附加业务字段；下游组件不得原地修改。
    """

    uid: str
    audio_path: Path
    label: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    speaker_id: str | None = None
    sample_rate_hint: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.uid, str) or not self.uid:
            raise ValueError(f"AudioRecord.uid 必须是非空字符串，实际: {self.uid!r}")
        if not isinstance(self.audio_path, Path):
            raise ValueError(
                f"AudioRecord.audio_path 必须是 pathlib.Path，实际: {type(self.audio_path)!r}"
            )
        if self.label is not None and not isinstance(self.label, int):
            raise ValueError(f"AudioRecord.label 必须是 int 或 None，实际: {self.label!r}")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError(
                f"AudioRecord.start_ms 必须 >= 0，实际: {self.start_ms} (uid={self.uid})"
            )
        if self.end_ms is not None:
            effective_start = self.start_ms or 0
            if self.end_ms <= effective_start:
                raise ValueError(
                    f"AudioRecord 要求 end_ms > start_ms（省略 start_ms 时按 0 处理），"
                    f"实际 start_ms={self.start_ms}, "
                    f"end_ms={self.end_ms} (uid={self.uid})"
                )
        if self.sample_rate_hint is not None and self.sample_rate_hint <= 0:
            raise ValueError(
                f"AudioRecord.sample_rate_hint 必须为正整数，实际: {self.sample_rate_hint}"
            )


@dataclass(frozen=True, slots=True)
class AudioData:
    """AudioLoader 的输出。

    约束（设计文档 §5.2）：

    - ``waveform`` 始终是二维 ``[C, T]``；单声道策略下 ``C == 1``，仍保留 channel 维。
    - 默认 dtype 为 ``torch.float32``。
    - Loader 不做 ``squeeze()``。
    """

    waveform: torch.Tensor
    sample_rate: int
    source_path: Path
    original_sample_rate: int
    num_frames: int

    def __post_init__(self) -> None:
        if self.waveform.dim() != 2:
            raise ValueError(
                f"AudioData.waveform 必须是 [C, T] 二维张量，实际 {self.waveform.dim()}D "
                f"{tuple(self.waveform.shape)}"
            )
        if self.waveform.shape[0] < 1 or self.waveform.shape[1] < 1:
            raise ValueError(
                f"AudioData.waveform 不允许空音频，实际 shape {tuple(self.waveform.shape)}"
            )
        if self.waveform.dtype != torch.float32:
            raise ValueError(
                f"AudioData.waveform dtype 必须是 float32，实际: {self.waveform.dtype}"
            )
        if self.sample_rate <= 0 or self.original_sample_rate <= 0:
            raise ValueError(
                f"AudioData 采样率必须为正，实际 sample_rate={self.sample_rate}, "
                f"original_sample_rate={self.original_sample_rate}"
            )
        if not torch.isfinite(self.waveform).all():
            raise ValueError("AudioData.waveform 包含 NaN/Inf")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """单个输入 tensor 的形状契约。

    Attributes:
        layout: 白名单 layout（``T`` / ``FT`` / ``TD`` / ``D`` / ``CFT``）。
        dtype: 期望 dtype。
        feature_dim: 固定特征维度（如 n_mels）；``None`` 表示不固定。
        time_axis: 时间轴索引；``None`` 表示非时序输入（如 ``D``）。
        pad_value: collate 时使用的 padding 值。
    """

    layout: str
    dtype: torch.dtype = torch.float32
    feature_dim: int | None = None
    time_axis: int | None = None
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        _validate_layout_name(self.layout)
        expected_axis = _LAYOUT_SHAPE_TABLE[self.layout][1]
        if self.time_axis is None:
            # 从 layout 推导标准时间轴（D layout 推导为 None）
            object.__setattr__(self, "time_axis", expected_axis)
        elif self.time_axis != expected_axis:
            raise ValueError(
                f"TensorSpec.time_axis={self.time_axis} 与 layout={self.layout} 的标准时间轴 "
                f"{expected_axis} 不一致；time_axis=None 表示非时序输入"
            )
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError(f"TensorSpec.feature_dim 必须为正，实际: {self.feature_dim}")
        if self.layout == LAYOUT_T and self.feature_dim is not None:
            raise ValueError("layout='T' 表示纯时间轴，不允许配置 feature_dim")

    @property
    def temporal(self) -> bool:
        """是否为时序输入。"""
        return is_temporal(self.layout)

    def validate_tensor(
        self,
        tensor: torch.Tensor,
        *,
        key: str,
        uid: str | None = None,
        error_cls: type[Exception] = RepresentationError,
    ) -> None:
        """校验单个 tensor 是否满足本 spec。

        Args:
            tensor: 待校验张量。
            key: 输入 key 名称。
            uid: 样本 uid（批处理上下文传 ``None``）。
            error_cls: 使用的异常类型；Dataset/Pipeline 上下文用 ``RepresentationError``，
                Collator 上下文用 ``CollationError``。
        """
        if tensor.dim() != _LAYOUT_SHAPE_TABLE[self.layout][0]:
            raise error_cls(
                f"输入 '{key}' 不满足 layout '{self.layout}': 期望 "
                f"{_LAYOUT_SHAPE_TABLE[self.layout][0]}D，实际 {tensor.dim()}D "
                f"{tuple(tensor.shape)}",
                uid=uid,
                component=key,
                stage="spec_validation",
            )
        if tensor.dtype != self.dtype:
            raise error_cls(
                f"输入 '{key}' dtype 不匹配: 期望 {self.dtype}，实际 {tensor.dtype}",
                uid=uid,
                component=key,
                stage="spec_validation",
            )
        if self.feature_dim is not None:
            dim = _feature_dim_index(self.layout)
            if tensor.shape[dim] != self.feature_dim:
                raise error_cls(
                    f"输入 '{key}' feature_dim 不匹配: 期望 {self.feature_dim}，"
                    f"实际 {tensor.shape[dim]} (shape {tuple(tensor.shape)})",
                    uid=uid,
                    component=key,
                    stage="spec_validation",
                )


def _feature_dim_index(layout: str) -> int:
    """返回 feature_dim 校验时检查的维度索引。"""
    return {
        LAYOUT_FT: 0,
        LAYOUT_TD: 1,
        LAYOUT_D: 0,
        LAYOUT_CFT: 1,
    }[layout]


def validate_representation_output(
    output: "RepresentationOutput",
    specs: Mapping[str, TensorSpec],
) -> None:
    """校验 RepresentationOutput 是否满足声明的 specs（设计文档 §5.4）。"""
    missing_specs = set(output.inputs) - set(specs)
    if missing_specs:
        raise RepresentationError(
            f"Representation 输出了未声明的 key: {sorted(missing_specs)}，"
            f"声明的 specs 为 {sorted(specs)}",
            component="representation",
            stage="contract_validation",
        )
    missing_inputs = set(specs) - set(output.inputs)
    if missing_inputs:
        raise RepresentationError(
            f"Representation 未输出声明的 key: {sorted(missing_inputs)}",
            component="representation",
            stage="contract_validation",
        )
    for key, tensor in output.inputs.items():
        specs[key].validate_tensor(tensor, key=key)

    unknown_lengths = set(output.lengths) - set(output.inputs)
    if unknown_lengths:
        raise RepresentationError(
            f"lengths 中的 key 必须属于 inputs，未知 key: {sorted(unknown_lengths)}",
            component="representation",
            stage="contract_validation",
        )
    for key, length in output.lengths.items():
        spec = specs[key]
        if not spec.temporal:
            raise RepresentationError(
                f"非时序输入 '{key}' (layout={spec.layout}) 不允许出现在 lengths 中",
                component="representation",
                stage="contract_validation",
            )
        tensor = output.inputs[key]
        axis = spec.time_axis
        actual = tensor.shape[axis]
        if length != actual:
            raise RepresentationError(
                f"lengths['{key}']={length} 与时间轴长度 {actual} 不一致 "
                f"(shape {tuple(tensor.shape)})",
                component="representation",
                stage="contract_validation",
            )
        if length <= 0:
            raise RepresentationError(
                f"lengths['{key}'] 必须为正，实际: {length}",
                component="representation",
                stage="contract_validation",
            )


@dataclass(frozen=True, slots=True)
class RepresentationOutput:
    """Representation 的输出。

    约束：

    - 单一波形表示统一使用 key ``waveform``；单一声学表示统一使用 key ``features``；
      多分支表示使用有语义的稳定 key。
    - 只有时序输入需要出现在 ``lengths``。
    """

    inputs: dict[str, torch.Tensor]
    lengths: dict[str, int]


@dataclass(frozen=True, slots=True)
class SERSample:
    """Dataset 的单样本输出。"""

    uid: str
    inputs: dict[str, torch.Tensor]
    lengths: dict[str, int]
    label: int | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.uid:
            raise ValueError("SERSample.uid 不能为空")
        unknown_lengths = set(self.lengths) - set(self.inputs)
        if unknown_lengths:
            raise ValueError(
                f"SERSample.lengths 中的 key 必须属于 inputs，未知 key: {sorted(unknown_lengths)}"
            )


@dataclass(frozen=True, slots=True)
class SERBatch:
    """Collator 的批次输出（设计文档 §5.6）。

    约束：

    - 分类标签 dtype 为 ``torch.long``。
    - 无标签推理 batch 的 ``labels`` 为 ``None``。
    - mask 使用 ``True`` 表示有效位置，``False`` 表示 padding。
    - ``window_map[i]`` 表示第 i 个滑窗来自原始 batch 的哪个样本。
    """

    inputs: dict[str, torch.Tensor]
    lengths: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    labels: torch.Tensor | None
    uids: list[str]
    metadata: list[dict[str, Any]]
    window_map: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.labels is not None and self.labels.dtype != torch.long:
            raise ValueError(f"SERBatch.labels dtype 必须是 torch.long，实际: {self.labels.dtype}")
        unknown_lengths = set(self.lengths) - set(self.inputs)
        if unknown_lengths:
            raise ValueError(
                f"SERBatch.lengths 中的 key 必须属于 inputs，未知 key: {sorted(unknown_lengths)}"
            )
        unknown_masks = set(self.masks) - set(self.inputs)
        if unknown_masks:
            raise ValueError(
                f"SERBatch.masks 中的 key 必须属于 inputs，未知 key: {sorted(unknown_masks)}"
            )
        if self.window_map is not None and self.labels is not None:
            if self.window_map.shape[0] != self.labels.shape[0]:
                raise ValueError(
                    f"window_map 行数 ({self.window_map.shape[0]}) 必须与 labels 数量 "
                    f"({self.labels.shape[0]}) 一致"
                )
        batch_sizes = {key: value.shape[0] for key, value in self.inputs.items()}
        if batch_sizes and len(set(batch_sizes.values())) != 1:
            raise ValueError(f"SERBatch.inputs 的 batch size 不一致: {batch_sizes}")
        batch_size = next(iter(batch_sizes.values()), 0)
        if self.labels is not None and self.labels.shape != (batch_size,):
            raise ValueError(
                f"SERBatch.labels 必须是 [B]，期望 ({batch_size},)，实际 {tuple(self.labels.shape)}"
            )
        if len(self.uids) != batch_size or len(self.metadata) != batch_size:
            raise ValueError(
                f"SERBatch.uids/metadata 必须与 batch 对齐: B={batch_size}, "
                f"uids={len(self.uids)}, metadata={len(self.metadata)}"
            )
        for key, value in self.lengths.items():
            if value.dtype != torch.long or value.shape != (batch_size,):
                raise ValueError(
                    f"SERBatch.lengths['{key}'] 必须是 shape=[B] 的 long tensor，"
                    f"实际 dtype={value.dtype}, shape={tuple(value.shape)}"
                )
        for key, value in self.masks.items():
            if value.dtype != torch.bool or value.dim() != 2 or value.shape[0] != batch_size:
                raise ValueError(
                    f"SERBatch.masks['{key}'] 必须是 shape=[B,T] 的 bool tensor，"
                    f"实际 dtype={value.dtype}, shape={tuple(value.shape)}"
                )
            if key in self.lengths and torch.any(self.lengths[key] > value.shape[1]):
                raise ValueError(f"SERBatch.lengths['{key}'] 超过 mask 的时间长度")
        if self.window_map is not None:
            if self.window_map.dtype != torch.long or self.window_map.shape != (batch_size,):
                raise ValueError(
                    f"window_map 必须是 shape=[B] 的 long tensor，实际 "
                    f"dtype={self.window_map.dtype}, shape={tuple(self.window_map.shape)}"
                )
            if torch.any(self.window_map < 0):
                raise ValueError("window_map 不允许包含负索引")


def validate_sample_contract(sample: SERSample, specs: Mapping[str, TensorSpec]) -> None:
    """运行时样本契约校验（debug/strict 模式调用，见设计文档 T3.4）。"""
    if set(sample.inputs) != set(specs):
        raise RepresentationError(
            f"样本 '{sample.uid}' 输入 key 与 specs 不一致: "
            f"实际 {sorted(sample.inputs)}，期望 {sorted(specs)}",
            uid=sample.uid,
            component="pipeline",
            stage="contract_validation",
        )
    for key, tensor in sample.inputs.items():
        specs[key].validate_tensor(tensor, key=key, uid=sample.uid)
    for key, length in sample.lengths.items():
        spec = specs[key]
        if not spec.temporal:
            raise RepresentationError(
                f"样本 '{sample.uid}' 非时序输入 '{key}' 不允许出现在 lengths",
                uid=sample.uid,
                component="pipeline",
                stage="contract_validation",
            )
        if length != sample.inputs[key].shape[spec.time_axis]:
            raise RepresentationError(
                f"样本 '{sample.uid}' lengths['{key}']={length} 与时间轴长度 "
                f"{sample.inputs[key].shape[spec.time_axis]} 不一致",
                uid=sample.uid,
                component="pipeline",
                stage="contract_validation",
            )
