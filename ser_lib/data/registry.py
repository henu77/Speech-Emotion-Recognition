"""组件注册表与组件描述符（设计文档 §12.2、§12.3）。

注册表按 ``namespace`` 隔离 importer / representation / transform 等组件族，
重复注册默认报错；组件参数由各自的 Pydantic 配置模型校验（``extra="forbid"``），
未知组件和未知参数都会在任务启动前失败，禁止静默降级。

桌面端通过 :class:`ComponentDescriptor` 枚举组件与生成表单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
from pydantic import BaseModel, ValidationError

from ser_lib.data.errors import RegistryError
from ser_lib.data.types import TensorSpec

# 组件状态
STATUS_STABLE = "stable"
STATUS_EXPERIMENTAL = "experimental"
STATUS_UNAVAILABLE = "unavailable"

# 桌面端普通组件列表只返回 stable
PUBLIC_STATUSES: tuple[str, ...] = (STATUS_STABLE,)


@dataclass(frozen=True)
class ComponentDescriptor:
    """暴露给桌面端的组件描述信息。

    ``config_schema`` 来自 Pydantic 配置模型的 JSON Schema；前端不得假设所有
    JSON Schema 特性都受支持（第一版限制到数字、字符串、布尔、枚举、数组和
    简单嵌套对象）。
    """

    id: str
    display_name: str
    category: str
    version: str = "1.0"
    status: str = STATUS_STABLE
    description: str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)
    input_specs: dict[str, TensorSpec] | None = None
    output_specs: dict[str, TensorSpec] | None = None

    def to_json_safe(self) -> dict[str, Any]:
        """序列化为 JSON 安全结构；``torch.dtype`` 转换为稳定字符串。"""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "category": self.category,
            "version": self.version,
            "status": self.status,
            "description": self.description,
            "config_schema": self.config_schema,
            "input_specs": _specs_to_json_safe(self.input_specs),
            "output_specs": _specs_to_json_safe(self.output_specs),
        }


def _dtype_to_str(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }
    if dtype not in mapping:
        raise RegistryError(f"不支持序列化的 dtype: {dtype!r}")
    return mapping[dtype]


def _spec_to_json_safe(spec: TensorSpec) -> dict[str, Any]:
    return {
        "layout": spec.layout,
        "dtype": _dtype_to_str(spec.dtype),
        "feature_dim": spec.feature_dim,
        "time_axis": spec.time_axis,
        "pad_value": spec.pad_value,
    }


def _specs_to_json_safe(specs: Mapping[str, TensorSpec] | None) -> dict[str, Any] | None:
    if specs is None:
        return None
    return {key: _spec_to_json_safe(value) for key, value in specs.items()}


@dataclass(frozen=True)
class ComponentEntry:
    """注册表中的一条组件记录。"""

    namespace: str
    name: str
    factory: Callable[..., Any]
    config_model: type[BaseModel] | None
    descriptor: ComponentDescriptor


class Registry:
    """命名空间隔离的组件注册表。

    用法::

        registry.register(
            namespace="representation",
            name="log_mel",
            factory=LogMelRepresentation,
            config_model=LogMelConfig,
            descriptor=ComponentDescriptor(id="log_mel", ...),
        )
        rep = registry.create("representation", {"type": "log_mel", "params": {...}})
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], ComponentEntry] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        namespace: str,
        name: str,
        factory: Callable[..., Any],
        config_model: type[BaseModel] | None = None,
        descriptor: ComponentDescriptor | None = None,
        replace: bool = False,
    ) -> None:
        """注册组件。

        Raises:
            RegistryError: ``(namespace, name)`` 重复且未显式 ``replace``；
                descriptor id 与注册名不一致；config_model 不可实例化。
        """
        if not namespace or not name:
            raise RegistryError(f"namespace 与 name 不能为空: {namespace!r}/{name!r}")
        key = (namespace, name)
        if key in self._entries and not replace:
            raise RegistryError(
                f"组件重复注册: namespace={namespace!r}, name={name!r}。"
                f"如需覆盖请显式传入 replace=True"
            )
        if descriptor is not None and descriptor.id != name:
            raise RegistryError(
                f"descriptor.id ({descriptor.id!r}) 必须与注册名称 ({name!r}) 一致"
            )
        if config_model is not None:
            try:
                config_model()
            except Exception as exc:  # noqa: BLE001 - 校验模型可用性
                # Pydantic ValidationError 说明模型本身有效（只是存在必填字段）；
                # 其他异常说明模型定义有问题，注册阶段即失败。
                if isinstance(exc, ValidationError):
                    pass
                else:
                    raise RegistryError(
                        f"组件 {namespace}.{name} 的 config_model 无法实例化: {exc}"
                    ) from exc
        if descriptor is None:
            descriptor = ComponentDescriptor(id=name, display_name=name, category=namespace)
        self._entries[key] = ComponentEntry(
            namespace=namespace, name=name, factory=factory,
            config_model=config_model, descriptor=descriptor,
        )

    # ------------------------------------------------------------------
    # 查询与创建
    # ------------------------------------------------------------------

    def get_entry(self, namespace: str, name: str) -> ComponentEntry:
        """获取组件记录，未知组件报错并列出可用项。"""
        key = (namespace, name)
        if key not in self._entries:
            available = sorted(n for (ns, n) in self._entries if ns == namespace)
            raise RegistryError(
                f"未知组件: namespace={namespace!r}, name={name!r}。"
                f"可用组件: {available or '（空）'}"
            )
        return self._entries[key]

    def create(self, namespace: str, component: Mapping[str, Any] | str, **overrides: Any) -> Any:
        """根据组件配置创建实例。

        Args:
            namespace: 命名空间。
            component: ``{"type": ..., "params": {...}}`` 或直接是类型名字符串。
            overrides: 追加以关键字形式传入工厂的参数（优先级高于 params）。

        Raises:
            RegistryError: 未知组件类型或参数校验失败。
        """
        if isinstance(component, str):
            comp_type, params = component, {}
        else:
            if not isinstance(component, Mapping):
                raise RegistryError(
                    f"组件配置必须是 Mapping 或 str，实际: {type(component)!r}"
                )
            unknown_keys = set(component) - {"type", "params"}
            if unknown_keys:
                raise RegistryError(
                    f"组件配置包含未知字段 {sorted(unknown_keys)}，仅允许 'type' 与 'params'"
                )
            comp_type = component.get("type")
            params = dict(component.get("params") or {})
            if not isinstance(comp_type, str) or not comp_type:
                raise RegistryError(f"组件 'type' 必须是非空字符串，实际: {comp_type!r}")
            if not isinstance(params, dict):
                raise RegistryError(
                    f"组件 'params' 必须是字典，实际: {type(params)!r}"
                )

        entry = self.get_entry(namespace, comp_type)
        kwargs: dict[str, Any] = {}
        if entry.config_model is not None:
            try:
                validated = entry.config_model(**params)
            except Exception as exc:  # noqa: BLE001 - 统一转业务异常
                raise RegistryError(
                    f"组件 {namespace}.{comp_type} 参数校验失败: {exc}",
                    component=comp_type,
                    stage="config_validation",
                ) from exc
            kwargs = validated.model_dump()
        else:
            kwargs = dict(params)
        kwargs.update(overrides)
        try:
            return entry.factory(**kwargs)
        except RegistryError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转业务异常
            raise RegistryError(
                f"组件 {namespace}.{comp_type} 构建失败: {exc}",
                component=comp_type,
                stage="component_build",
            ) from exc

    # ------------------------------------------------------------------
    # 枚举（桌面端）
    # ------------------------------------------------------------------

    def names(self, namespace: str) -> list[str]:
        """列出命名空间下全部组件名。"""
        return sorted(n for (ns, n) in self._entries if ns == namespace)

    def descriptors(
        self,
        namespace: str,
        *,
        statuses: tuple[str, ...] = PUBLIC_STATUSES,
    ) -> list[ComponentDescriptor]:
        """枚举命名空间下的组件描述符。

        普通组件列表只返回 stable；experimental 组件需要显式传入 statuses。
        """
        result = []
        for (ns, name), entry in sorted(self._entries.items()):
            if ns != namespace:
                continue
            if entry.descriptor.status in statuses:
                result.append(entry.descriptor)
        return result

    def json_safe_descriptors(
        self,
        namespace: str,
        *,
        statuses: tuple[str, ...] = PUBLIC_STATUSES,
    ) -> list[dict[str, Any]]:
        """枚举命名空间下的组件描述符并序列化为 JSON 安全结构。"""
        return [d.to_json_safe() for d in self.descriptors(namespace, statuses=statuses)]


# 模块级默认注册表。库内组件全部注册在此；如需隔离可自行实例化 Registry。
default_registry = Registry()
