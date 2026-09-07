"""模型注册表。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel
from ser_lib.data.errors import RegistryError
from ser_lib.models.base import SERModel

@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    display_name: str
    description: str
    config_schema: dict[str, Any]
    input_layouts: dict[str, str]

    def to_json_safe(self) -> dict[str, Any]:
        return {
            "id": self.id, "display_name": self.display_name,
            "description": self.description, "config_schema": self.config_schema,
            "input_layouts": self.input_layouts,
        }

@dataclass(frozen=True)
class _ModelEntry:
    factory: Callable[..., SERModel]
    config_model: type[BaseModel] | None
    descriptor: ModelDescriptor

class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _ModelEntry] = {}

    def register(self, name: str, factory: Callable[..., SERModel], *,
                 config_model: type[BaseModel] | None = None,
                 descriptor: ModelDescriptor | None = None,
                 replace: bool = False) -> None:
        if not name:
            raise RegistryError("模型名称不能为空")
        if name in self._entries and not replace:
            raise RegistryError(f"模型重复注册: {name!r}")
        descriptor = descriptor or ModelDescriptor(name, name, "", {}, {})
        if descriptor.id != name:
            raise RegistryError("模型 descriptor.id 必须与注册名称一致")
        self._entries[name] = _ModelEntry(factory, config_model, descriptor)

    def create(self, name: str, **params: Any) -> SERModel:
        if name not in self._entries:
            raise RegistryError(f"未知模型 {name!r}，可用模型: {sorted(self._entries)}")
        entry = self._entries[name]
        try:
            if entry.config_model is not None:
                params = entry.config_model(**params).model_dump()
            model = entry.factory(**params)
        except Exception as exc:
            raise RegistryError(f"模型 {name!r} 构建失败: {exc}") from exc
        if not isinstance(model, SERModel):
            raise RegistryError(f"模型工厂 {name!r} 返回了非 SERModel: {type(model)!r}")
        return model

    def names(self) -> list[str]:
        return sorted(self._entries)

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._entries[name].descriptor.to_json_safe() for name in self.names()]

model_registry = ModelRegistry()
