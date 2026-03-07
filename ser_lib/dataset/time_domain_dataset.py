import torch
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset
from ser_lib.dataset.augment.builder import build_time_domain_transforms
from ser_lib.dataset.features.builder import build_feature_extractors


class TimeDomainDataset(BaseConfigDataset):
    """时域特征数据集。

    任务定位: 加载原始音频 → 时域增强 → 提取时域特征（F0、RMS、ZCR 等）
    → 将所有 1D 特征序列在特征维上拼接为单一矩阵返回。

    配置文件: time_domain_dataset.yaml

    __getitem__ 返回的 features 字典::

        {
            "time_features": Tensor[n_features, T]
        }

    Notes:
        各特征提取器必须使用相同的 hop_length 以保证时间步对齐。
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
        self.extractors = build_feature_extractors(features_cfg.get('time_domain', {}))

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        # 1. 时域增强
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 2. 逐特征提取并拼接
        feat_tensors = []
        for extractor in self.extractors.values():
            feat = extractor(waveform).squeeze(0)    # [T]
            feat_tensors.append(feat.unsqueeze(0))   # [1, T]

        if not feat_tensors:
            raw = waveform.squeeze(0)
            return {"time_features": raw.unsqueeze(0)}, raw.shape[-1]

        time_features = torch.cat(feat_tensors, dim=0)  # [n_features, T]
        return {"time_features": time_features}, time_features.shape[-1]
