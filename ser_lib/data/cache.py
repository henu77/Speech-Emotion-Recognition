"""确定性 Representation 的磁盘缓存装饰器。"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any
import torch
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import AudioData, RepresentationOutput, TensorSpec, validate_representation_output

logger = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 1

def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

class CachedRepresentation(Representation):
    """缓存任意确定性 Representation 的输出。

    为保证片段和上游确定性处理不会碰撞，缓存键包含 waveform 内容哈希。
    不建议包装随机增强后的表示；即使内容哈希保证正确，也会制造大量低命中缓存。
    """
    descriptor = ComponentDescriptor(
        id="cached", display_name="表示缓存", category="representation",
        description="确定性表示的本地磁盘缓存装饰器。",
    )

    def __init__(self, representation: Representation, directory: Path | str) -> None:
        super().__init__()
        self.representation = representation
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return self.representation.output_specs

    def cache_key(self, audio: AudioData) -> str:
        source_stat: dict[str, int] = {}
        try:
            stat = audio.source_path.stat()
            source_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            pass
        config = getattr(self.representation, "config", None)
        identity = {
            "schema": CACHE_SCHEMA_VERSION,
            "source": str(audio.source_path.resolve()),
            "source_stat": source_stat,
            "sample_rate": audio.sample_rate,
            "original_sample_rate": audio.original_sample_rate,
            "num_frames": audio.num_frames,
            "representation": self.representation.descriptor.id,
            "version": self.representation.descriptor.version,
            "config": _json_value(config),
        }
        digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        waveform = audio.waveform.detach().to("cpu").contiguous()
        digest.update(str(waveform.dtype).encode("ascii"))
        digest.update(str(tuple(waveform.shape)).encode("ascii"))
        digest.update(waveform.numpy().tobytes())
        return digest.hexdigest()

    def forward(self, audio: AudioData) -> RepresentationOutput:
        key = self.cache_key(audio)
        path = self.directory / key[:2] / f"{key}.pt"
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                    raise ValueError("缓存 schema 版本不匹配")
                output = RepresentationOutput(
                    inputs=dict(payload["inputs"]),
                    lengths={k: int(v) for k, v in payload["lengths"].items()},
                )
                validate_representation_output(output, self.output_specs)
                return output
            except Exception as exc:
                logger.warning("缓存条目损坏，将重新计算: %s (%s)", path, exc)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        output = self.representation(audio)
        validate_representation_output(output, self.output_specs)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        torch.save({
            "schema_version": CACHE_SCHEMA_VERSION,
            "inputs": {k: v.detach().cpu() for k, v in output.inputs.items()},
            "lengths": dict(output.lengths),
        }, temporary)
        try:
            temporary.replace(path)
        except OSError:
            # 并发 worker 可能已经写入同一内容；保留已存在的最终条目。
            temporary.unlink(missing_ok=True)
            if not path.is_file():
                raise
        return output

    def entry_count(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.pt"))

    def size_bytes(self) -> int:
        total = 0
        for path in self.directory.glob("*/*.pt"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total
