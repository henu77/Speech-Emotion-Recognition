"""通用 Collator：根据 TensorSpec 批处理，不根据 Dataset 类型分支（§11）。

第一版策略：

- ``dynamic``: 每个时序 key 按 batch 内最大长度 padding，生成 bool mask
  （True=有效，False=padding）；``D`` 类直接 stack，不生成 length/mask；
- ``fixed``: 按 key 配置最大长度，超长截断、不足 padding，输出截断后的
  有效长度，不生成 mask；
- ``sliding``: 滑动窗口，第一版仅支持一个主时序输入；输出 ``window_map``
  显式映射窗口 → 原始样本；最后一个不足窗口长度的片段 padding，
  length 为实际有效长度；输入恰好等于窗口长度时只产生一个窗口。

Mixup 属于 batch transform，在其契约（软标签表示、loss/metrics 协议、与
滑窗的组合规则）确定前保持 unavailable，不提供运行时占位分支（§11.3）。
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch
import torch.nn.functional as F

from ser_lib.data.config import BatchingConfig
from ser_lib.data.errors import CollationError
from ser_lib.data.types import (
    LAYOUT_TD,
    SERBatch,
    SERSample,
    TensorSpec,
    time_axis_of,
)


class CollateStrategy(str, Enum):
    """批处理策略（与 BatchingConfig.type 对齐）。"""

    DYNAMIC = "dynamic"
    FIXED = "fixed"
    SLIDING = "sliding"


class SERCollator:
    """基于输入规格的通用 Collator。

    Args:
        specs: ``{key: TensorSpec}``，通常来自 ``pipeline.output_specs``。
        batching: 批处理配置。
    """

    def __init__(self, specs: dict[str, TensorSpec], batching: BatchingConfig) -> None:
        batching.validate_completeness()
        self.specs = dict(specs)
        self.batching = batching
        self.strategy = CollateStrategy(batching.type)

        temporal_keys = [k for k, s in self.specs.items() if s.temporal]
        if self.strategy is CollateStrategy.FIXED:
            missing = [k for k in temporal_keys if k not in batching.fixed.max_lengths]
            if missing:
                raise CollationError(
                    f"fixed 策略要求为每个时序 key 配置 max_lengths，缺失: {missing}；"
                    f"时序 keys: {temporal_keys}",
                    component="collator", stage="collator_build",
                )
        if self.strategy is CollateStrategy.SLIDING:
            if len(temporal_keys) != 1:
                raise CollationError(
                    f"sliding 策略第一版仅支持一个主时序输入，实际时序 keys: "
                    f"{temporal_keys}。多输入滑窗需要先定义同步切窗语义",
                    component="collator", stage="collator_build",
                )
            if batching.primary_key is not None and batching.primary_key != temporal_keys[0]:
                raise CollationError(
                    f"batching.primary_key ({batching.primary_key!r}) 与唯一的时序 key "
                    f"({temporal_keys[0]!r}) 不一致",
                    component="collator", stage="collator_build",
                )

    # ------------------------------------------------------------------

    def __call__(self, samples: Sequence[SERSample]) -> SERBatch:
        samples = list(samples)
        if not samples:
            raise CollationError(
                "batch 为空，无法处理", component="collator", stage="collate"
            )

        expected_keys = set(self.specs)
        for sample in samples:
            if set(sample.inputs) != expected_keys:
                raise CollationError(
                    f"样本 '{sample.uid}' 输入 key 与 collator specs 不一致: "
                    f"实际 {sorted(sample.inputs)}，期望 {sorted(expected_keys)}",
                    uid=sample.uid, component="collator", stage="collate",
                )
            for key, tensor in sample.inputs.items():
                self.specs[key].validate_tensor(
                    tensor, key=key, error_cls=CollationError
                )

        labels: torch.Tensor | None
        if all(s.label is None for s in samples):
            labels = None
        elif any(s.label is None for s in samples):
            bad = [s.uid for s in samples if s.label is None]
            raise CollationError(
                f"一个 batch 内不允许部分样本有标签、部分没有；无标签样本: {bad}",
                component="collator", stage="collate",
            )
        else:
            labels = torch.tensor([int(s.label) for s in samples], dtype=torch.long)

        if self.strategy is CollateStrategy.DYNAMIC:
            return self._collate_dynamic(samples, labels)
        if self.strategy is CollateStrategy.FIXED:
            return self._collate_fixed(samples, labels)
        return self._collate_sliding(samples, labels)

    # ------------------------------------------------------------------
    # dynamic padding
    # ------------------------------------------------------------------

    def _collate_dynamic(self, samples: list[SERSample],
                         labels: torch.Tensor | None) -> SERBatch:
        inputs: dict[str, torch.Tensor] = {}
        lengths: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}

        for key, spec in self.specs.items():
            tensors = [s.inputs[key] for s in samples]
            if not spec.temporal:
                inputs[key] = torch.stack(tensors)
                continue
            lengths_list = [s.lengths[key] for s in samples]
            max_len = max(lengths_list)
            padded = [_pad_time(t, spec, max_len) for t in tensors]
            inputs[key] = torch.stack(padded)
            lengths[key] = torch.tensor(lengths_list, dtype=torch.long)
            masks[key] = torch.arange(max_len).expand(len(samples), max_len) < torch.tensor(
                lengths_list, dtype=torch.long
            ).unsqueeze(1)

        return SERBatch(
            inputs=inputs, lengths=lengths, masks=masks, labels=labels,
            uids=[s.uid for s in samples],
            metadata=[dict(s.metadata) for s in samples],
        )

    # ------------------------------------------------------------------
    # fixed length
    # ------------------------------------------------------------------

    def _collate_fixed(self, samples: list[SERSample],
                       labels: torch.Tensor | None) -> SERBatch:
        inputs: dict[str, torch.Tensor] = {}
        lengths: dict[str, torch.Tensor] = {}

        for key, spec in self.specs.items():
            tensors = [s.inputs[key] for s in samples]
            if not spec.temporal:
                inputs[key] = torch.stack(tensors)
                continue
            max_len = self.batching.fixed.max_lengths[key]
            adjusted = [_pad_time(_truncate_time(t, spec, max_len), spec, max_len)
                        for t in tensors]
            inputs[key] = torch.stack(adjusted)
            lengths[key] = torch.tensor(
                [min(s.lengths[key], max_len) for s in samples], dtype=torch.long
            )

        return SERBatch(
            inputs=inputs, lengths=lengths, masks={}, labels=labels,
            uids=[s.uid for s in samples],
            metadata=[dict(s.metadata) for s in samples],
        )

    # ------------------------------------------------------------------
    # sliding window
    # ------------------------------------------------------------------

    def _collate_sliding(self, samples: list[SERSample],
                         labels: torch.Tensor | None) -> SERBatch:
        spec_key = next(k for k, s in self.specs.items() if s.temporal)
        spec = self.specs[spec_key]
        window = self.batching.sliding.window_size
        stride = self.batching.sliding.stride

        all_windows: list[torch.Tensor] = []
        window_map: list[int] = []
        window_lengths: list[int] = []

        for sample_index, sample in enumerate(samples):
            tensor = sample.inputs[spec_key]
            total = sample.lengths[spec_key]
            if total <= window:
                # 短输入也至少产生一个窗口；恰好等于窗口长度时只产生一个窗口
                all_windows.append(_pad_time(tensor, spec, window))
                window_map.append(sample_index)
                window_lengths.append(total)
                continue

            # 完整窗口
            last_full = ((total - window) // stride) * stride
            for start in range(0, last_full + 1, stride):
                all_windows.append(_slice_time(tensor, spec, start, start + window))
                window_map.append(sample_index)
                window_lengths.append(window)

            # 最后一个不足窗口长度的片段：padding，length 为实际有效长度。
            # 若 (total - window) 恰为 stride 的整数倍，最后一个完整窗口
            # 已覆盖到结尾，不再产生尾窗。
            tail_start = last_full + stride
            if tail_start < total and (total - window) % stride != 0:
                tail = _slice_time(tensor, spec, tail_start, total)
                all_windows.append(_pad_time(tail, spec, window))
                window_map.append(sample_index)
                window_lengths.append(total - tail_start)

        expanded_labels: torch.Tensor | None = None
        if labels is not None:
            expanded_labels = labels[torch.tensor(window_map, dtype=torch.long)]

        # 非时序输入（例如 utterance-level 全局向量）按 window_map 复制，
        # 保证滑窗 batch 不丢失模型需要的辅助输入。
        expanded_inputs: dict[str, torch.Tensor] = {spec_key: torch.stack(all_windows)}
        map_tensor = torch.tensor(window_map, dtype=torch.long)
        for key, other_spec in self.specs.items():
            if key == spec_key:
                continue
            if other_spec.temporal:
                # 构造阶段已经保证只有一个时序 key；此分支仅作防御。
                raise CollationError(
                    f"sliding 策略不能处理额外时序输入 '{key}'",
                    component="collator", stage="collate",
                )
            stacked = torch.stack([sample.inputs[key] for sample in samples])
            expanded_inputs[key] = stacked[map_tensor]

        expanded_uids = [samples[index].uid for index in window_map]
        expanded_metadata = [dict(samples[index].metadata) for index in window_map]

        return SERBatch(
            inputs=expanded_inputs,
            lengths={spec_key: torch.tensor(window_lengths, dtype=torch.long)},
            masks={},
            labels=expanded_labels,
            uids=expanded_uids,
            metadata=expanded_metadata,
            window_map=map_tensor,
        )


# =====================================================================
# 时间轴操作工具
# =====================================================================


def _pad_time(tensor: torch.Tensor, spec: TensorSpec, target: int) -> torch.Tensor:
    """沿 spec 的时间轴右侧 padding 到 target 长度。"""
    axis = time_axis_of(spec.layout)
    current = tensor.shape[axis]
    if current >= target:
        return tensor
    pad_amount = target - current
    ndim = tensor.dim()
    pad = [0] * (2 * ndim)
    pad[-(2 * axis + 1)] = pad_amount
    return F.pad(tensor, tuple(pad), value=spec.pad_value)


def _truncate_time(tensor: torch.Tensor, spec: TensorSpec, target: int) -> torch.Tensor:
    """沿 spec 的时间轴截断到 target 长度。"""
    axis = time_axis_of(spec.layout)
    if tensor.shape[axis] <= target:
        return tensor
    if spec.layout == LAYOUT_TD:
        return tensor[:target]
    return tensor[..., :target]


def _slice_time(tensor: torch.Tensor, spec: TensorSpec, start: int, end: int) -> torch.Tensor:
    """沿 spec 的时间轴切片 [start, end)。"""
    axis = time_axis_of(spec.layout)
    if spec.layout == LAYOUT_TD:
        return tensor[start:end]
    return tensor[..., start:end]


def build_collator(specs: dict[str, TensorSpec], batching: BatchingConfig) -> SERCollator:
    """便捷构造函数。"""
    return SERCollator(specs, batching)
