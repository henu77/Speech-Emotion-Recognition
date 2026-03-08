import torch
import torchaudio.transforms as T
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset
from ser_lib.dataset.augment.builder import build_time_domain_transforms, build_freq_domain_transforms
from ser_lib.dataset.features.builder import build_feature_extractor


class SpectrogramDataset(BaseConfigDataset):
    """谱图数据集。

    任务定位: 支持提取指定的谱图 (Spectrogram, MelSpectrogram, LogMelSpectrogram, MFCC)。
    对指定的单一谱图应用频域增强并以字典形式返回。
    配置文件: spectrogram_dataset.yaml

    __getitem__ 返回的 features 字典::

        {
            "melspectrogram": Tensor[Freq, T],
            ... # 取决于选择的类型
        }
    """

    def __init__(self, config_path: str, split: str = 'train'):
        super().__init__(config_path, split)

        transforms_cfg = self.config.get('transforms', {})
        spec_cfg = self.config.get('spectrogram', {})
        
        # 用户指定的谱图类型
        self.spec_type_name = spec_cfg.get('type', 'MelSpectrogram')
        spec_kwargs = spec_cfg.get('kwargs', {})

        self.wave_transform = build_time_domain_transforms(
            transforms_cfg.get('waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )
        self.adv_wave_transform = build_time_domain_transforms(
            transforms_cfg.get('advanced_waveform_level', {}).get(split, []),
            sample_rate=self.target_sr,
        )
        self.spec_transform = build_freq_domain_transforms(
            transforms_cfg.get('spectrogram_level', {}).get(split, [])
        )

        # 构建谱图提取器
        self.extractor = build_feature_extractor(self.spec_type_name, spec_kwargs, self.target_sr)

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        # 1. （可选）时域增强
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 2. 谱图提取
        feat = self.extractor(waveform)

        # 3. 频域增强
        if self.spec_transform:
            feat = self.spec_transform(feat)

        feat = feat.squeeze(0)  # [Freq, T]
        
        return {self.spec_type_name.lower(): feat}, feat.shape[-1]
