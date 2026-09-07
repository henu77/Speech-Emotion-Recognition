"""整个 SER 基础库共享的异常类型。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SERError(Exception):
    """所有可预期 SER 领域错误的根类型。

    ``code`` 是稳定的机器可读标识，``str(exc)`` 仍保持面向人的消息。
    ``details`` 只保存可安全序列化的诊断信息，不应放入密钥或大对象。
    """

    default_code = "ser_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("异常 message 必须是非空字符串")
        resolved_code = code or self.default_code
        if not resolved_code or not resolved_code.replace("_", "").isalnum():
            raise ValueError(f"异常 code 非法: {resolved_code!r}")
        super().__init__(message)
        self.code = resolved_code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """返回适合 CLI、日志或其他调用方消费的结构化信息。"""
        return {"code": self.code, "message": str(self), "details": dict(self.details)}


class ConfigurationError(SERError):
    """配置文件无法读取、版本不兼容或内容校验失败。"""

    default_code = "configuration_error"


class OperationCancelled(SERError):
    """调用方请求取消一个可取消操作。"""

    default_code = "operation_cancelled"
