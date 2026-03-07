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
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        raw = waveform.squeeze(0)  # [T]
        return {"raw_waveform": raw}, raw.shape[-1]
