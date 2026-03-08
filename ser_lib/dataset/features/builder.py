import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Dict, Any

from ser_lib.dataset.features.time_domain import (
    PitchF0, RMS, ZeroCrossingRate, JitterShimmerHNR
)

from ser_lib.dataset.features.freq_domain import (
    SpectralCentroid, SpectralRolloff, SpectralFlatness, SpectralFlux, ComputeDeltas
)

def build_feature_extractor(feat_type: str, cfg: Dict[str, Any], target_sr: int = 16000) -> nn.Module:
    """
    统一构建单一特征提取器。可被 FeatureDataset 或 SpectrogramDataset 各自单独调用。
    """
    cfg = cfg.copy() if cfg else {}

    # 需要传入采样率的特征白名单（自动兜底补全 target_sr）
    needs_sr = {
        "MelSpectrogram", "LogMelSpectrogram", "MFCC", 
        "F0", "JitterShimmerHNR", "SpectralCentroid", "SpectralRolloff"
    }
    if feat_type in needs_sr:
        cfg.setdefault('sample_rate', target_sr)

    # -------------------------------------------------------------
    # 经典谱图与倒谱
    # -------------------------------------------------------------
    if feat_type == "Spectrogram":
        return T.Spectrogram(**cfg)
        
    elif feat_type == "MelSpectrogram":
        return T.MelSpectrogram(**cfg)

    elif feat_type == "LogMelSpectrogram":
        melspec = T.MelSpectrogram(**cfg)
        amp_to_db = T.AmplitudeToDB()
        return nn.Sequential(melspec, amp_to_db)
        
    elif feat_type == "MFCC":
        # 提取平铺的 mel 参数组装成 melkwargs 以适配 torchaudio.transforms.MFCC
        mel_keys = ['n_fft', 'win_length', 'hop_length', 'f_min', 'f_max', 'n_mels', 'center', 'pad', 'pad_mode']
        melkwargs = cfg.pop('melkwargs', {})
        for k in list(cfg.keys()):
            if k in mel_keys:
                melkwargs[k] = cfg.pop(k)
        if melkwargs:
            cfg['melkwargs'] = melkwargs
            
        return T.MFCC(**cfg)
        
    # -------------------------------------------------------------
    # 韵律与时域
    # -------------------------------------------------------------
    elif feat_type == "F0":
        return PitchF0(**cfg)
        
    elif feat_type == "RMS":
        return RMS(**cfg)
        
    elif feat_type == "ZCR":
        return ZeroCrossingRate(**cfg)
        
    elif feat_type == "JitterShimmerHNR":
        return JitterShimmerHNR(**cfg)
        
    # -------------------------------------------------------------
    # 频域时变与滚降参数
    # -------------------------------------------------------------
    elif feat_type == "SpectralCentroid":
        return SpectralCentroid(**cfg)
        
    elif feat_type == "SpectralRolloff":
        return SpectralRolloff(**cfg)
        
    elif feat_type == "SpectralFlatness":
        return SpectralFlatness(**cfg)
        
    elif feat_type == "SpectralFlux":
        return SpectralFlux(**cfg)
        
    # -------------------------------------------------------------
    # 动态差分
    # -------------------------------------------------------------
    elif feat_type == "Delta":
        return ComputeDeltas(**cfg)
        
    else:
        raise ValueError(f"未知的特征提取器类型: {feat_type}")
