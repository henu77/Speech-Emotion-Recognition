"""不会擅自配置宿主应用 root logger 的日志工具。"""

from __future__ import annotations

import logging
from typing import TextIO


LOGGER_NAME = "ser_lib"


def get_logger(name: str | None = None) -> logging.Logger:
    """获取库命名空间下的 logger。"""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(name if name.startswith(f"{LOGGER_NAME}.") else f"{LOGGER_NAME}.{name}")


def configure_library_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
) -> logging.Handler:
    """显式为 ``ser_lib`` 安装一个 handler，并返回它供调用方移除。

    重复调用不会清除调用方已经安装的 handler，也不会修改 root logger。
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.addHandler(handler)
    return handler
