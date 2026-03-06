import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Dict, Any

from ser_lib.dataset.features.time_domain import (
    PitchF0, RMS, ZeroCrossingRate, JitterShimmerHNR
)

from ser_lib.dataset.features.freq_domain import (
    MFCC, SpectralCentroid, SpectralRolloff, SpectralFlatness, SpectralFlux, ComputeDeltas
)

def build_feature_extractors(features_cfg: Dict[str, Any]) -> nn.ModuleDict:
    """
    根据配置文件构建多模态并行特征提取模块映射表。
    
    支持基频、短时能量、过零率、MFCC、语谱图、Log-Mel 以及一系列进阶的频域动态特征。
    将所有配置中请求的提取器打包成 ModuleDict 以便 `base_dataset.py` 中独立调用。
    """
    extractors = nn.ModuleDict()
    
    if not features_cfg:
        return extractors
        
    for feat_name, cfg in features_cfg.items():
        feat_type = cfg.get('type')
        
        # -------------------------------------------------------------
        # 原有经典 2D 频谱组
        # -------------------------------------------------------------
        if feat_type == "Spectrogram":
            extractors[feat_name] = T.Spectrogram(
                n_fft=cfg.get('n_fft', 1024),
                win_length=cfg.get('win_length'),
                hop_length=cfg.get('hop_length', 256),
                power=cfg.get('power', 2.0)
            )
            
        elif feat_type == "MelSpectrogram":
            extractors[feat_name] = T.MelSpectrogram(
                sample_rate=cfg.get('sample_rate', 16000),
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256),
                n_mels=cfg.get('n_mels', 80),
                f_min=cfg.get('f_min', 0.0),
                f_max=cfg.get('f_max', None)
            )

        elif feat_type == "LogMelSpectrogram":
            # LogMel 只是对 Mel 取对数
            # 返回: Sequential( MelSpectrogram, AmplitudeToDB )
            melspec = T.MelSpectrogram(
                sample_rate=cfg.get('sample_rate', 16000),
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256),
                n_mels=cfg.get('n_mels', 80),
                f_min=cfg.get('f_min', 0.0),
                f_max=cfg.get('f_max', None)
            )
            amp_to_db = T.AmplitudeToDB(stype='power', top_db=cfg.get('top_db', 80.0))
            extractors[feat_name] = nn.Sequential(melspec, amp_to_db)
            
        # -------------------------------------------------------------
        # 新增韵律与频域 1D/2D 拓展组
        # -------------------------------------------------------------
        elif feat_type == "MFCC":
            extractors[feat_name] = MFCC(
                sample_rate=cfg.get('sample_rate', 16000),
                n_mfcc=cfg.get('n_mfcc', 13),
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256),
                n_mels=cfg.get('n_mels', 80)
            )
            
        elif feat_type == "F0":
            extractors[feat_name] = PitchF0(
                sample_rate=cfg.get('sample_rate', 16000),
                hop_length=cfg.get('hop_length', 256)
            )
            
        elif feat_type == "RMS":
            extractors[feat_name] = RMS(
                win_length=cfg.get('win_length', 400),
                hop_length=cfg.get('hop_length', 256)
            )
            
        elif feat_type == "ZCR":
            extractors[feat_name] = ZeroCrossingRate(
                win_length=cfg.get('win_length', 400),
                hop_length=cfg.get('hop_length', 256)
            )
            
        # --- 频域拓展 ---
        elif feat_type == "SpectralCentroid":
            extractors[feat_name] = SpectralCentroid(
                sample_rate=cfg.get('sample_rate', 16000),
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256)
            )
            
        elif feat_type == "SpectralRolloff":
            extractors[feat_name] = SpectralRolloff(
                sample_rate=cfg.get('sample_rate', 16000),
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256),
                roll_percent=cfg.get('roll_percent', 0.85)
            )
            
        elif feat_type == "SpectralFlatness":
            extractors[feat_name] = SpectralFlatness(
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256)
            )
            
        elif feat_type == "SpectralFlux":
            extractors[feat_name] = SpectralFlux(
                n_fft=cfg.get('n_fft', 1024),
                hop_length=cfg.get('hop_length', 256)
            )
            
        # -------------------------------------------------------------
        # 高阶装甲层：动态附加
        # -------------------------------------------------------------
        elif feat_type == "Delta":
            # 这里是一个代理提取器：它需要在提取时额外输入底层矩阵序列
            # 请在 collate_fn 或具体使用时额外使用 `features['mfcc_delta'] = ComputeDeltas()(features['mfcc'])` 这样的调用
            extractors[feat_name] = ComputeDeltas(win_length=cfg.get('win_length', 5))
            
        # --------------
        else:
            raise ValueError(f"未知的特征提取器类型或名字配置错误: {feat_type}")
            
    return extractors
