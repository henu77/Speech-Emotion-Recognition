"""AudioLoader：音频解码、片段读取、声道转换与重采样（设计文档 §7）。

执行顺序（§7.2）::

    解析并验证最终路径 → 读取音频元信息 → 毫秒片段转 frame offset
    → 只读取目标片段 → 校验非空且有限值 → 声道转换 → 重采样
    → 可选确定性归一化 → 返回 [C, T] float32

Loader 不做 ``squeeze()``、不了解 label、不理解 split，也不负责 GPU 搬运。
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from ser_lib.data.errors import (
    AudioDecodeError,
    AudioNotFoundError,
    InvalidAudioSegmentError,
    SERDataError,
)
from ser_lib.data.types import AudioData, AudioRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioLoaderConfig:
    """AudioLoader 运行时配置（设计文档 §7.1）。"""

    target_sample_rate: int = 16000
    mono: bool = True
    normalize_peak: bool = False
    backend: str = "torchaudio"


class AudioLoader:
    """从 :class:`AudioRecord` 加载音频并输出标准化的 :class:`AudioData`。

    重采样器按 ``(orig_sr, target_sr, dtype, device)`` 缓存（§7.3）；
    实例可被 DataLoader worker pickle，不在 worker 间共享不可序列化状态。
    """

    def __init__(self, config: AudioLoaderConfig | None = None) -> None:
        config = config or AudioLoaderConfig()
        if config.backend != "torchaudio":
            raise SERDataError(
                f"不支持的音频后端: {config.backend!r}，当前仅支持 'torchaudio'"
            )
        if config.target_sample_rate <= 0:
            raise SERDataError(
                f"target_sample_rate 必须为正，实际: {config.target_sample_rate}"
            )
        self.config = config
        # 缓存 key: (orig_freq, new_freq, dtype, device)
        self._resamplers: dict[tuple[int, int, torch.dtype, torch.device], T.Resample] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load(self, record: AudioRecord, *, base_dir: Path | None = None) -> AudioData:
        """加载一条记录对应的音频。

        Args:
            record: 音频记录；相对路径基于 ``base_dir``（通常是 manifest root）解析。
            base_dir: 相对路径解析基准目录。

        Raises:
            AudioNotFoundError: 文件不存在。
            AudioDecodeError: 解码失败。
            InvalidAudioSegmentError: 片段非法或解码结果为空。
        """
        path = self.resolve_path(record.audio_path, base_dir)
        uid = record.uid

        if not path.exists():
            raise AudioNotFoundError(
                "音频文件不存在", uid=uid, path=path, component="audio_loader",
                stage="resolve",
            )
        if not path.is_file():
            raise AudioDecodeError(
                "音频路径不是文件", uid=uid, path=path, component="audio_loader",
                stage="resolve",
            )

        try:
            info = torchaudio.info(str(path))
        except Exception as exc:  # noqa: BLE001 - torchaudio 抛出的异常类型因后端而异
            raise AudioDecodeError(
                "读取音频元信息失败", uid=uid, path=path,
                component="audio_loader", stage="probe",
            ) from exc

        original_sr = int(info.sample_rate)
        total_frames = int(info.num_frames)

        frame_offset, num_frames = self._segment_to_frames(
            record, original_sr=original_sr, total_frames=total_frames, uid=uid, path=path,
        )

        try:
            waveform, sr = torchaudio.load(
                str(path), frame_offset=frame_offset, num_frames=num_frames,
            )
        except Exception as exc:  # noqa: BLE001
            raise AudioDecodeError(
                "音频解码失败", uid=uid, path=path, component="audio_loader",
                stage="decode",
            ) from exc

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            raise InvalidAudioSegmentError(
                "解码结果为空音频（0 帧）", uid=uid, path=path,
                component="audio_loader", stage="decode",
            )
        if sr != original_sr:
            # 极少数后端会返回与元信息不同的采样率
            logger.warning(
                "音频实际采样率 (%s) 与元信息 (%s) 不一致: uid=%s, path=%s",
                sr, original_sr, uid, path,
            )
            original_sr = int(sr)

        if not torch.isfinite(waveform).all():
            raise AudioDecodeError(
                "音频包含 NaN/Inf，拒绝加载（不做自动替换）", uid=uid, path=path,
                component="audio_loader", stage="validate",
            )

        # 声道转换：默认求均值；mono 开启时输出 [1, T]
        if self.config.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样至目标采样率
        if sr != self.config.target_sample_rate:
            waveform = self._resample(waveform, sr)

        # 可选的确定性 loader 级归一化
        if self.config.normalize_peak:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        waveform = waveform.to(torch.float32)

        if waveform.shape[-1] == 0:
            raise InvalidAudioSegmentError(
                "重采样后音频为空", uid=uid, path=path,
                component="audio_loader", stage="resample",
            )

        return AudioData(
            waveform=waveform,
            sample_rate=self.config.target_sample_rate,
            source_path=path,
            original_sample_rate=original_sr,
            num_frames=int(waveform.shape[-1]),
        )

    @staticmethod
    def resolve_path(audio_path: Path, base_dir: Path | None) -> Path:
        """解析音频路径：绝对路径直接规范化，相对路径基于 base_dir。"""
        path = Path(audio_path)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return path.resolve()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _segment_to_frames(
        self,
        record: AudioRecord,
        *,
        original_sr: int,
        total_frames: int,
        uid: str,
        path: Path,
    ) -> tuple[int, int]:
        """把毫秒片段转换为 frame offset / num_frames。"""
        start_ms = record.start_ms
        end_ms = record.end_ms
        if start_ms is None and end_ms is None:
            return 0, -1

        if start_ms is None:
            start_ms = 0

        frame_offset = int(round(start_ms / 1000.0 * original_sr))
        if frame_offset >= total_frames > 0:
            raise InvalidAudioSegmentError(
                f"片段起点超出音频长度: start_ms={start_ms} 对应帧 {frame_offset}，"
                f"音频总帧数 {total_frames}",
                uid=uid, path=path, component="audio_loader", stage="segment",
            )

        if end_ms is None:
            # 负数 num_frames 表示读到文件结尾
            return frame_offset, -1

        end_frame = int(round(end_ms / 1000.0 * original_sr))
        if end_frame > total_frames > 0:
            warnings.warn(
                f"片段 end_ms={end_ms} 超出音频长度（总帧数 {total_frames}），"
                f"将截断到文件结尾: uid={uid}, path={path}",
                UserWarning,
                stacklevel=2,
            )
        num_frames = max(end_frame - frame_offset, 0)
        return frame_offset, num_frames

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """用缓存的 resampler 重采样。key 含 (orig, target, dtype, device)。"""
        key = (orig_sr, self.config.target_sample_rate, waveform.dtype, waveform.device)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.config.target_sample_rate)
            self._resamplers[key] = resampler
        return resampler(waveform)
