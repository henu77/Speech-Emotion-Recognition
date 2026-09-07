"""模型兼容性契约（设计文档 §13）。

``validate_compatibility`` 在创建训练任务前校验表示输出、批处理配置与模型
输入要求；兼容性错误必须在训练启动前返回，不得等到第一个 forward 才通过
shape error 暴露。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ser_lib.data.config import BatchingConfig
from ser_lib.data.errors import CompatibilityError
from ser_lib.data.types import TensorSpec


@dataclass(frozen=True)
class ModelSpec:
    """模型输入规格（模型显式声明，禁止调用方猜测 tensor shape）。"""

    model_id: str
    required_inputs: dict[str, TensorSpec]
    supports_masks: bool
    supports_variable_length: bool
    num_classes: int | None
    expected_sample_rate: int | None = None


def validate_compatibility(
    representation_specs: Mapping[str, TensorSpec],
    model_spec: ModelSpec,
    batching_config: BatchingConfig,
    *,
    num_classes: int | None = None,
    sample_rate: int | None = None,
) -> None:
    """校验表示 specs、批处理配置与模型 spec 的兼容性。

    Raises:
        CompatibilityError: 任一检查失败；错误信息包含全部失败原因。
    """
    problems: list[str] = []

    # 1. 所有 required input key 必须存在
    missing = sorted(set(model_spec.required_inputs) - set(representation_specs))
    if missing:
        problems.append(
            f"表示未提供模型必需的输入 key: {missing}（表示输出: {sorted(representation_specs)}）"
        )

    for key, required in model_spec.required_inputs.items():
        actual = representation_specs.get(key)
        if actual is None:
            continue
        # 2. layout 完全匹配；第一版不自动转置
        if actual.layout != required.layout:
            problems.append(
                f"输入 '{key}' layout 不匹配: 模型要求 {required.layout}，"
                f"表示输出 {actual.layout}（第一版不自动转置）"
            )
        # 3. 固定 feature dimension 匹配
        if required.feature_dim is not None and actual.feature_dim is not None \
                and required.feature_dim != actual.feature_dim:
            problems.append(
                f"输入 '{key}' feature_dim 不匹配: 模型要求 {required.feature_dim}，"
                f"表示输出 {actual.feature_dim}"
            )
        # 4. 可变长度输入需要模型支持 mask（dynamic padding 依赖 mask）
        if (
            actual.temporal
            and batching_config.type == "dynamic"
            and not model_spec.supports_masks
        ):
            problems.append(
                f"输入 '{key}' 为可变长度时序输入且批处理策略为 dynamic，"
                f"但模型 {model_spec.model_id} 不支持 mask"
            )

    # 5. 不支持可变长度的模型必须配置固定长度 Collator
    if not model_spec.supports_variable_length:
        if batching_config.type != "fixed":
            problems.append(
                f"模型 {model_spec.model_id} 不支持可变长度输入，"
                f"必须配置 batching.type='fixed'，实际: {batching_config.type}"
            )
        else:
            assert batching_config.fixed is not None
            temporal_keys = [k for k, s in representation_specs.items() if s.temporal]
            missing_lengths = [
                k for k in temporal_keys
                if k not in batching_config.fixed.max_lengths
            ]
            if missing_lengths:
                problems.append(
                    f"模型要求固定长度输入，但 fixed.max_lengths 缺少时序 key: "
                    f"{missing_lengths}"
                )

    # 6. 类别数量一致
    if num_classes is not None and model_spec.num_classes is not None \
            and num_classes != model_spec.num_classes:
        problems.append(
            f"类别数不一致: 数据集 {num_classes} 类，模型 {model_spec.model_id} "
            f"输出 {model_spec.num_classes} 类"
        )

    if model_spec.expected_sample_rate is not None and sample_rate is not None \
            and model_spec.expected_sample_rate != sample_rate:
        problems.append(
            f"采样率不一致: 模型 {model_spec.model_id} 要求 "
            f"{model_spec.expected_sample_rate} Hz，数据流水线输出 {sample_rate} Hz"
        )

    # 7. 滑动窗口产生数量可变的窗口，模型需要支持可变批次语义
    if batching_config.type == "sliding" and not model_spec.supports_variable_length:
        problems.append(
            f"批处理策略为 sliding（每批窗口数量可变），但模型 "
            f"{model_spec.model_id} 不支持可变长度输入；"
            f"滑窗推理还需要显式的窗口聚合策略"
        )

    if problems:
        raise CompatibilityError(
            "模型兼容性校验失败:\n- " + "\n- ".join(problems),
            component="compatibility_check",
            stage="task_startup",
        )
