import torch
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset
from ser_lib.dataset.augment.builder import build_time_domain_transforms, build_freq_domain_transforms
from ser_lib.dataset.features.builder import build_feature_extractors


class FreqDomainDataset(BaseConfigDataset):
    """频域特征数据集。

    任务定位: 加载原始音频 → 可选时域增强 → 提取频域特征（Log-Mel、MFCC 等 2D 时频谱）
    → 对每张频谱图独立应用频域增强 → 以字典形式分别返回各频域特征。

    配置文件: freq_domain_dataset.yaml

    __getitem__ 返回的 features 字典::

        {
            "log_mel": Tensor[n_mels, T],
            "mfcc":    Tensor[n_mfcc, T],   # 若同时开启
            ...
        }
    """

    def __init__(self, config_path: str, split: str = 'train'):
        super().__init__(config_path, split)

        transforms_cfg = self.config.get('transforms', {})
        features_cfg = self.config.get('features', {})

        self.wave_transform = build_time_domain_transforms(
            transforms_cfg.get('waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )
        self.adv_wave_transform = build_time_domain_transforms(
            transforms_cfg.get('advanced_waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )
        self.extractors = build_feature_extractors(features_cfg.get('freq_domain', {}))
        self.spec_transform = build_freq_domain_transforms(
            transforms_cfg.get('spectrogram_level', {}).get(split, [])
        )

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        # 1. （可选）时域增强
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 2. 逐特征提取 + 频域增强（每张谱图独立）
        features: Dict[str, torch.Tensor] = {}
        for feat_name, extractor in self.extractors.items():
            feat = extractor(waveform)
            feat = self.spec_transform(feat)
            features[feat_name] = feat.squeeze(0)   # [Freq, T]

        if not features:
            return {}, 0

        seq_length = list(features.values())[0].shape[-1]
        return features, seq_length
