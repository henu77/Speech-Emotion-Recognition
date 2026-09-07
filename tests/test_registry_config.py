import pytest
from pydantic import BaseModel, ConfigDict

from ser_lib.data.errors import RegistryError
from ser_lib.data.registry import ComponentDescriptor, Registry


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 1


class Component:
    def __init__(self, value=1):
        self.value = value


def _registry():
    registry = Registry()
    registry.register(
        namespace="test",
        name="component",
        factory=Component,
        config_model=StrictConfig,
        descriptor=ComponentDescriptor(id="component", display_name="Component", category="test"),
    )
    return registry


def test_registry_rejects_duplicate_registration():
    registry = _registry()
    with pytest.raises(RegistryError, match="重复注册"):
        registry.register(namespace="test", name="component", factory=Component)


def test_registry_rejects_unknown_component_parameter():
    registry = _registry()
    with pytest.raises(RegistryError, match="参数校验失败"):
        registry.create("test", {"type": "component", "params": {"typo": 2}})


def test_registry_descriptor_is_json_safe():
    descriptor = _registry().json_safe_descriptors("test")[0]
    assert descriptor["id"] == "component"
