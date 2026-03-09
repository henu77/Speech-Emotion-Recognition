import torch
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset
from ser_lib.dataset.augment.builder import build_time_domain_transforms


class WaveformDataset(BaseConfigDataset):
    """原始波形数据集。

    任务定位: 仅加载原始音频波形并应用时域数据增强，不做任何特征提取。
    适用于端到端原始波形输入的模型，如 Wav2Vec2、HuBERT、RawNet 等。

    配置文件: waveform_dataset.yaml

    __getitem__ 返回的 features 字典::

        {
            "raw_waveform": Tensor[T]   # 一维波形，已去除 channel 维
        }
    """

    def __init__(self, config_path: str, split: str = 'train'):
        super().__init__(config_path, split)

        transforms_cfg = self.config.get('transforms', {})
        self.wave_transform = build_time_domain_transforms(
            transforms_cfg.get('waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )
        self.adv_wave_transform = build_time_domain_transforms(
            transforms_cfg.get('advanced_waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        # 维度检查：输入波形应为 [1, T] 或 [C, T]
        assert waveform.dim() == 2, (
            f"[WaveformDataset._load_item] 输入波形维度错误: 期望 2D [1, T], 实际 {waveform.dim()}D {tuple(waveform.shape)}"
        )

        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 维度检查：增强后波形仍应为 2D
        assert waveform.dim() == 2, (
            f"[WaveformDataset._load_item] 增强后波形维度错误: 期望 2D [1, T], 实际 {waveform.dim()}D {tuple(waveform.shape)}"
        )

        raw = waveform.squeeze(0)  # [T]

        # 维度检查：输出波形应为 1D [T]
        assert raw.dim() == 1, (
            f"[WaveformDataset._load_item] 输出波形维度错误: 期望 1D [T], 实际 {raw.dim()}D {tuple(raw.shape)}"
        )

        seq_length = raw.shape[-1]

        # 维度检查：seq_length 与波形长度一致性
        assert seq_length == raw.shape[-1], (
            f"[WaveformDataset._load_item] seq_length 不一致: seq_length={seq_length}, 波形长度={raw.shape[-1]}"
        )

        return {"raw_waveform": raw}, seq_length
