import torch
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset
from ser_lib.dataset.augment.builder import build_time_domain_transforms
from ser_lib.dataset.features.builder import build_feature_extractor
import torch.nn as nn


class FeatureDataset(BaseConfigDataset):
    """特征域数据集。

    任务定位: 用户能自由选择特征 (时域等)，按词典直接返回，不强制拼接。
    配置文件: feature_dataset.yaml

    __getitem__ 返回的 features 字典::

        {
            "feature_name_1": Tensor[T_1] 或 Tensor[D, T_1],
            "feature_name_2": Tensor[T_2] 或 Tensor[D, T_2],
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
        self.extractors = nn.ModuleDict()
        for feat_name, cfg in features_cfg.get('selected_features', {}).items():
            feat_type = cfg.get('type')
            kwargs = cfg.get('kwargs', {})
            self.extractors[feat_name] = build_feature_extractor(feat_type, kwargs, self.target_sr)

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        # 1. 时域增强
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 2. 逐特征提取并放入字典
        features: Dict[str, torch.Tensor] = {}
        for feat_name, extractor in self.extractors.items():
            feat = extractor(waveform).squeeze(0)
            features[feat_name] = feat

        if not features:
            raw = waveform.squeeze(0)
            return {"raw_waveform": raw}, raw.shape[-1]

        seq_length = list(features.values())[0].shape[-1]
        return features, seq_length
