"""SER 数据模块异常层级。

层级结构（设计文档 §7.4）::

    SERDataError
    ├── ManifestError
    ├── AudioNotFoundError
    ├── AudioDecodeError
    ├── InvalidAudioSegmentError
    ├── RepresentationError
    ├── CollationError
    └── CompatibilityError

异常消息必须包含 ``uid`` 与解析后的音频路径（如适用）；底层异常保留为
``__cause__``，保证错误可定位。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SERDataError(Exception):
    """数据模块所有业务异常的基类。"""

    def __init__(self, message: str, *, uid: str | None = None,
                 path: Path | str | None = None,
                 component: str | None = None,
                 stage: str | None = None) -> None:
        """初始化异常。

        Args:
            message: 人类可读的错误描述。
            uid: 出错样本的记录 ID（如有）。
            path: 解析后的音频/文件路径（如有）。
            component: 出错组件名称，如 ``log_mel``。
            stage: 失败阶段，如 ``decode`` / ``representation`` / ``collate``。
        """
        parts: list[str] = []
        if uid is not None:
            parts.append(f"uid={uid}")
        if path is not None:
            parts.append(f"path={Path(path)}")
        if component is not None:
            parts.append(f"component={component}")
        if stage is not None:
            parts.append(f"stage={stage}")
        if parts:
            message = f"{message} [{'; '.join(parts)}]"
        super().__init__(message)
        self.uid = uid
        self.path = Path(path) if path is not None else None
        self.component = component
        self.stage = stage


class ManifestError(SERDataError):
    """Manifest 读取、校验或路径解析失败。"""


class AudioNotFoundError(SERDataError):
    """音频文件不存在。"""


class AudioDecodeError(SERDataError):
    """音频解码失败或内容损坏。"""


class InvalidAudioSegmentError(SERDataError):
    """音频片段定义非法（越界、零长度或解码结果为空）。"""


class RepresentationError(SERDataError):
    """表示（Representation）计算失败或输出违反契约。"""


class TransformError(SERDataError):
    """Transform 构建或执行失败。"""


class CollationError(SERDataError):
    """批处理（collate）失败：key 不一致、layout 不匹配、部分样本缺标签等。"""


class CompatibilityError(SERDataError):
    """表示、批处理与模型输入要求之间的兼容性校验失败。"""


class RegistryError(SERDataError):
    """注册表操作失败：重复注册、未知组件、schema 校验失败等。"""


def wrap_error(
    exc: Exception,
    target: type[SERDataError],
    message: str,
    *,
    uid: str | None = None,
    path: Any = None,
    component: str | None = None,
    stage: str | None = None,
) -> SERDataError:
    """把底层异常包装为业务异常并保留 ``__cause__``。

    若 ``exc`` 已经是目标类型则原样返回，避免重复包装丢失上下文。
    """
    if isinstance(exc, target):
        return exc
    wrapped = target(message, uid=uid, path=path, component=component, stage=stage)
    wrapped.__cause__ = exc
    return wrapped
