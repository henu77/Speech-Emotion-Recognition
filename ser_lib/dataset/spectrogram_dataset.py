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
        # 维度检查：输入波形应为 [1, T] 或 [C, T]
        assert waveform.dim() == 2, (
            f"[SpectrogramDataset._load_item] 输入波形维度错误: 期望 2D [1, T], 实际 {waveform.dim()}D {tuple(waveform.shape)}"
        )

        # 1. （可选）时域增强
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)

        # 维度检查：增强后波形仍应为 2D
        assert waveform.dim() == 2, (
            f"[SpectrogramDataset._load_item] 增强后波形维度错误: 期望 2D [1, T], 实际 {waveform.dim()}D {tuple(waveform.shape)}"
        )

        # 2. 谱图提取
        feat = self.extractor(waveform)

        # 维度检查：谱图提取后应为 3D [1, Freq, T] 或 [C, Freq, T]
        assert feat.dim() == 3, (
            f"[SpectrogramDataset._load_item] 谱图提取后维度错误: 期望 3D [1, Freq, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
        )

        # 3. 频域增强
        if self.spec_transform:
            feat = self.spec_transform(feat)

        # 维度检查：频域增强后仍应为 3D
        assert feat.dim() == 3, (
            f"[SpectrogramDataset._load_item] 频域增强后维度错误: 期望 3D [1, Freq, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
        )

        feat = feat.squeeze(0)  # [Freq, T]

        # 维度检查：输出谱图应为 2D [Freq, T]
        assert feat.dim() == 2, (
            f"[SpectrogramDataset._load_item] 输出谱图维度错误: 期望 2D [Freq, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
        )

        seq_length = feat.shape[-1]

        # 维度检查：seq_length 与时间帧数一致性
        assert seq_length == feat.shape[-1], (
            f"[SpectrogramDataset._load_item] seq_length 不一致: seq_length={seq_length}, 时间帧数={feat.shape[-1]}"
        )

        return {self.spec_type_name.lower(): feat}, seq_length
