"""SamplePipeline：串联波形 transform → Representation → 特征 transform。

（设计文档 §10.2）Pipeline 输出 :class:`SERSample`；训练与离线推理共用
同一套构建函数（``build_pipeline``），保证预处理一致。
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch.nn as nn

if TYPE_CHECKING:
    from ser_lib.data.audio import AudioLoader

from ser_lib.data.config import ComponentConfig, DataConfig
from ser_lib.data.errors import RepresentationError, TransformError
from ser_lib.data.registry import default_registry
from ser_lib.data.representations.base import Representation
from ser_lib.data.transforms.base import (
    FeatureTransformPipeline,
    RandomApply,
    WaveformTransformPipeline,
    validate_feature_transform_layouts,
)
from ser_lib.data.types import (
    AudioData,
    AudioRecord,
    RepresentationOutput,
    SERSample,
    TensorSpec,
    validate_representation_output,
)


class SamplePipeline(nn.Module):
    """单样本处理流水线。

    - ``waveform_transforms`` 可为 ``None``（验证/测试/推理：无随机增强）；
    - ``representation`` 必填；
    - ``feature_transforms`` 可为 ``None``；
    - ``validate_contract`` 开启时对每个输出做契约校验（测试与开发默认开启，
      生产可关闭以提升吞吐）。
    """

    def __init__(
        self,
        representation: Representation,
        waveform_transforms: nn.Module | None = None,
        feature_transforms: nn.Module | None = None,
        *,
        validate_contract: bool = True,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.waveform_transforms = waveform_transforms
        self.feature_transforms = feature_transforms
        self.validate_contract = validate_contract

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        """输出形状契约（与 representation 一致）。"""
        return self.representation.output_specs

    def forward(self, audio: AudioData, record: AudioRecord) -> SERSample:
        waveform = audio.waveform
        if self.waveform_transforms is not None:
            try:
                waveform = self.waveform_transforms(waveform)
            except Exception as exc:  # noqa: BLE001
                raise TransformError(
                    f"波形 transform 失败: {exc}",
                    uid=record.uid, path=audio.source_path,
                    component="waveform_transforms", stage="transform",
                ) from exc
            if waveform.dim() != 2:
                raise TransformError(
                    f"波形 transform 输出必须是 [C, T]，实际 "
                    f"{waveform.dim()}D {tuple(waveform.shape)}",
                    uid=record.uid, path=audio.source_path,
                    component="waveform_transforms", stage="transform",
                )
            audio = replace(
                audio, waveform=waveform, num_frames=int(waveform.shape[-1])
            )

        try:
            output = self.representation(audio)
        except RepresentationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RepresentationError(
                f"表示计算失败: {exc}",
                uid=record.uid, path=audio.source_path,
                component=self.representation.descriptor.id, stage="representation",
            ) from exc

        if self.feature_transforms is not None:
            try:
                new_inputs = self.feature_transforms(dict(output.inputs))
            except Exception as exc:  # noqa: BLE001
                raise TransformError(
                    f"特征 transform 失败: {exc}",
                    uid=record.uid, path=audio.source_path,
                    component="feature_transforms", stage="transform",
                ) from exc
            lengths = {}
            for key, tensor in new_inputs.items():
                spec = self.representation.output_specs[key]
                if spec.temporal:
                    lengths[key] = int(tensor.shape[spec.time_axis])
            output = RepresentationOutput(inputs=new_inputs, lengths=lengths)

        if self.validate_contract:
            validate_representation_output(output, self.representation.output_specs)

        metadata = dict(record.metadata)
        if record.speaker_id is not None:
            metadata.setdefault("speaker_id", record.speaker_id)

        return SERSample(
            uid=record.uid,
            inputs=dict(output.inputs),
            lengths=dict(output.lengths),
            label=record.label,
            metadata=metadata,
        )

    # nn.Module __call__ 会转发参数；这里显式声明类型签名
    def __call__(self, audio: AudioData, record: AudioRecord) -> SERSample:  # noqa: F811
        return super().__call__(audio, record)


# =====================================================================
# 配置 → 组件构建（训练与推理共享）
# =====================================================================


def build_representation(component: Mapping[str, Any] | str) -> Representation:
    """根据组件配置从注册表构建表示。"""
    if isinstance(component, Mapping) and not isinstance(component, str):
        component = {"type": component["type"], "params": dict(component.get("params") or {})}
    rep = default_registry.create("representation", component)
    if not isinstance(rep, Representation):
        raise RepresentationError(
            f"注册表返回的对象不是 Representation: {type(rep)!r}",
            stage="component_build",
        )
    return rep


def _factory_accepts(factory: type, param: str) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover
        return False
    return param in signature.parameters


def _build_waveform_transforms(
    configs: Sequence[ComponentConfig],
    *,
    sample_rate: int,
    allow_random: bool,
) -> WaveformTransformPipeline | None:
    """构建波形 transform 流水线。

    随机 transform 在 ``allow_random=False``（val/test/predict）时整体跳过；
    确定性 transform（如 normalize）始终应用。
    """
    modules: list[nn.Module] = []
    for component in configs:
        factory, _, _ = _waveform_transform_entry(component.type)
        if _factory_accepts(factory, "sample_rate") and "sample_rate" not in component.params:
            # 按 AudioLoader 的目标采样率注入（如 pitch_shift）
            module = factory(**{**component.params, "sample_rate": sample_rate})
        else:
            module = default_registry.create(
                "waveform_transform",
                {"type": component.type, "params": component.params},
            )
        if getattr(factory, "is_random", True):
            if not allow_random:
                continue
            probability = component.probability if component.probability is not None else 0.5
            modules.append(RandomApply(module, probability))
        else:
            if component.probability is not None:
                raise ValueError(
                    f"确定性 transform '{component.type}' 不允许配置 probability"
                )
            modules.append(module)
    if not modules:
        return None
    return WaveformTransformPipeline(modules)


def _waveform_transform_entry(name: str):
    from ser_lib.data.transforms.waveform import WAVEFORM_TRANSFORM_SPECS

    try:
        return WAVEFORM_TRANSFORM_SPECS[name]
    except KeyError:
        from ser_lib.data.errors import RegistryError

        raise RegistryError(
            f"未知波形 transform: {name!r}，可用: {sorted(WAVEFORM_TRANSFORM_SPECS)}"
        ) from None


def _build_feature_transforms(
    configs: Sequence[ComponentConfig],
    *,
    specs: dict[str, TensorSpec],
    allow_random: bool,
) -> FeatureTransformPipeline | None:
    """构建特征 transform 流水线，并做构建期 layout 兼容性校验。"""
    modules: list[nn.Module] = []
    for component in configs:
        if component.type != "spec_masking":
            from ser_lib.data.errors import RegistryError

            raise RegistryError(
                f"未知特征 transform: {component.type!r}，可用: ['spec_masking']"
            )
        module = default_registry.create(
            "feature_transform", {"type": component.type, "params": component.params}
        )
        validate_feature_transform_layouts(module, specs)
        if not allow_random:
            continue
        probability = component.probability if component.probability is not None else 0.5
        modules.append(RandomApply(module, probability))
    if not modules:
        return None
    return FeatureTransformPipeline(modules)


def build_pipeline(
    data_config: DataConfig,
    *,
    train: bool,
    validate_contract: bool = True,
) -> SamplePipeline:
    """从 DataConfig 构建 SamplePipeline（训练与推理共享同一入口）。

    Args:
        data_config: 数据配置。
        train: True 时启用配置中的随机增强；False（验证/测试/推理）默认
            禁用随机增强（设计文档 §8.3）。
        validate_contract: 是否做运行时输出契约校验。
    """
    representation = build_representation(data_config.representation.model_dump())
    if data_config.cache.enabled:
        if train and data_config.waveform_transforms:
            raise ValueError(
                "训练随机波形增强之后默认禁止 Representation 缓存；"
                "请关闭 data.cache 或移除训练 waveform_transforms"
            )
        from ser_lib.data.cache import CachedRepresentation

        representation = CachedRepresentation(
            representation, data_config.cache.directory
        )
    waveform_transforms = _build_waveform_transforms(
        data_config.waveform_transforms,
        sample_rate=data_config.audio.target_sample_rate,
        allow_random=train,
    )
    feature_transforms = _build_feature_transforms(
        data_config.feature_transforms,
        specs=representation.output_specs,
        allow_random=train,
    )
    return SamplePipeline(
        representation=representation,
        waveform_transforms=waveform_transforms,
        feature_transforms=feature_transforms,
        validate_contract=validate_contract,
    )


def build_components(
    data_config: DataConfig,
    *,
    train: bool,
    validate_contract: bool = True,
) -> tuple["AudioLoader", SamplePipeline]:
    """构建 (AudioLoader, SamplePipeline) 组件对。"""
    from ser_lib.data.audio import AudioLoader, AudioLoaderConfig

    loader_config = AudioLoaderConfig(
        target_sample_rate=data_config.audio.target_sample_rate,
        mono=data_config.audio.mono,
        normalize_peak=data_config.audio.normalize_peak,
        backend=data_config.audio.backend,
    )
    loader = AudioLoader(loader_config)
    pipeline = build_pipeline(data_config, train=train, validate_contract=validate_contract)
    return loader, pipeline
