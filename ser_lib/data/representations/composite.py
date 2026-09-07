"""CompositeRepresentation：组合多个子表示，支持不等长时间轴（设计文档 §9.4）。

默认不对齐不同输出；不同子表示可以返回不同时间分辨率的帧级特征与
utterance 级全局向量。只有用户显式选择 TemporalAligner 时才执行时间对齐
（第一版未实现，数据结构已保留不等长多输入）。
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from ser_lib.data.errors import RepresentationError
from ser_lib.data.registry import ComponentDescriptor, default_registry
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import (
    AudioData,
    RepresentationOutput,
    TensorSpec,
)


class CompositeConfig(BaseModel):
    """组合表示参数：``{key: 组件配置}``。"""

    model_config = ConfigDict(extra="forbid")

    outputs: dict[str, dict[str, Any]] = Field(..., min_length=1)


class CompositeRepresentation(Representation):
    """组合多个子表示。

    子表示输出 key 重命名规则：

    - 单输出 key 的子表示（``features`` 或 ``waveform``）重命名为组合配置中的
      目标 key；
    - 多输出 key 的子表示（如 AcousticFeatures 的帧级 + 全局音质向量）中，
      ``features`` 重命名为目标 key，其余 key 保留原名（已具备语义）。

    key 冲突、未知组件都会在构建阶段失败（验证优先于运行）。
    """

    descriptor = ComponentDescriptor(
        id="composite",
        display_name="组合表示",
        category="representation",
        description="组合多个子表示，支持不同时间分辨率与 utterance 级全局向量，"
                    "默认不做时间对齐。",
        config_schema=CompositeConfig.model_json_schema(),
    )

    def __init__(self, **params: Any) -> None:
        config = CompositeConfig(**params)
        super().__init__()
        self.config = config

        self._sub_representations: dict[str, Representation] = {}
        # target_key -> sub-representation
        self._key_mapping: dict[str, tuple[str, Representation]] = {}
        for target_key, sub_config in config.outputs.items():
            sub_rep = default_registry.create("representation", sub_config)
            if not isinstance(sub_rep, Representation):
                raise RepresentationError(
                    f"组合表示的子组件必须是 Representation，实际: {type(sub_rep)!r}",
                    component=self.descriptor.id,
                    stage="component_build",
                )
            self._sub_representations[target_key] = sub_rep

        nn_map: dict[str, Representation] = {}
        self._modules_map = nn_map
        for target_key, sub_rep in self._sub_representations.items():
            sub_keys = set(sub_rep.output_specs.keys())
            if len(sub_keys) == 1:
                renamed = {target_key: next(iter(sub_keys))}
            elif "features" in sub_keys:
                renamed = {target_key: "features"}
                renamed.update({k: k for k in sub_keys - {"features"}})
            else:
                renamed = {k: k for k in sub_keys}
            for new_key, old_key in renamed.items():
                if new_key in self._key_mapping:
                    raise RepresentationError(
                        f"组合表示输出 key 冲突: '{new_key}' 被多个子表示占用",
                        component=self.descriptor.id,
                        stage="component_build",
                    )
                self._key_mapping[new_key] = (old_key, sub_rep)
            nn_map[target_key] = sub_rep
        self.sub_representations = torch.nn.ModuleDict(nn_map)

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        specs: dict[str, TensorSpec] = {}
        for new_key, (old_key, sub_rep) in self._key_mapping.items():
            specs[new_key] = sub_rep.output_specs[old_key]
        return specs

    def forward(self, audio: AudioData) -> RepresentationOutput:
        inputs: dict[str, torch.Tensor] = {}
        lengths: dict[str, int] = {}
        for sub_rep in self._sub_representations.values():
            out = sub_rep(audio)
            for new_key, (old_key, owner) in self._key_mapping.items():
                if owner is not sub_rep or old_key not in out.inputs:
                    continue
                inputs[new_key] = out.inputs[old_key]
                if old_key in out.lengths:
                    lengths[new_key] = out.lengths[old_key]
        return RepresentationOutput(inputs=inputs, lengths=lengths)
