import torch
from ser_lib.dataset.features.builder import build_feature_extractors

class TimeDomainFeatureLoader:
    """提取时域 1D 特征（如 F0, RMS, ZCR）"""
    def __init__(self, features_cfg: dict):
        self.extractors = build_feature_extractors(features_cfg)
        
    def __call__(self, waveform: torch.Tensor) -> dict:
        features = {}
        for feat_name, extractor in self.extractors.items():
            feat = extractor(waveform)
            # Remove dummy batch dimension
            features[feat_name] = feat.squeeze(0)
        return features
