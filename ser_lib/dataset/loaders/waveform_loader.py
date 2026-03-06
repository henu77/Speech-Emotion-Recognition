import torch

class WaveformLoader:
    """提取原始波形数据"""
    def __init__(self, features_cfg: dict):
        self.features_cfg = features_cfg
        
    def __call__(self, waveform: torch.Tensor) -> dict:
        features = {}
        if not self.features_cfg:
            return features
            
        # 如果配置项里要求了提取原始波形，例如 raw_waveform: {type: "Raw"}
        for feat_name, cfg in self.features_cfg.items():
            if cfg.get('type') == 'Raw':
                # 去除批次维度
                features[feat_name] = waveform.squeeze(0)
        return features
