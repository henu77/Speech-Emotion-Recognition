import torch
from ser_lib.dataset.features.builder import build_feature_extractors
from ser_lib.dataset.augment.builder import build_freq_domain_transforms

class FreqDomainFeatureLoader:
    """提取频域 2D 特征（如 Mel, MFCC 等）并应用频谱级数据增强"""
    def __init__(self, features_cfg: dict, augment_cfg: list):
        self.extractors = build_feature_extractors(features_cfg)
        self.spec_transform = build_freq_domain_transforms(augment_cfg)
        
    def __call__(self, waveform: torch.Tensor) -> dict:
        features = {}
        for feat_name, extractor in self.extractors.items():
            feat = extractor(waveform)
            
            # Apply spec augmentations if it is a spectrogram
            feat = self.spec_transform(feat)
            
            # Remove dummy batch dimension
            features[feat_name] = feat.squeeze(0)
        return features
