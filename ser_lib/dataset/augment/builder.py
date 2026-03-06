import torch
from ser_lib.dataset.augment.time_domain import (
    AddGaussianNoise, PitchShift, TimeStretch, TimeShift, 
    VolumeScale, RIRSimulation, DynamicSNRMixing
)
from ser_lib.dataset.augment.freq_domain import (
    SpecMasking, FilterAugment, VTLP, SpecMix
)

def build_time_domain_transforms(transforms_cfg: list, sample_rate: int) -> torch.nn.Module:
    """根据配置字典数组实例化指定的波形增强变换为整体流水线 Module。"""
    if not transforms_cfg:
        return torch.nn.Identity()

    pipeline = []
    for cfg in transforms_cfg:
        t_type = cfg.get('type')
        p = cfg.get('p', 0.5)
        
        if t_type == "Normalize":
            # 无参即时归一化算子
            class Normalize(torch.nn.Module):
                def forward(self, w): return (w - w.mean()) / (w.std() + 1e-8)
            pipeline.append(Normalize())
            
        elif t_type == "AddGaussianNoise":
            pipeline.append(AddGaussianNoise(snr=cfg.get('snr', 15.0), p=p))
        elif t_type == "PitchShift":
            pipeline.append(PitchShift(sample_rate=sample_rate, n_steps=cfg.get('n_steps', 4), p=p))
        elif t_type == "TimeStretch":
            pipeline.append(TimeStretch(rate=cfg.get('rate', 1.2), p=p))
        elif t_type == "TimeShift":
            pipeline.append(TimeShift(shift_max_ratio=cfg.get('shift_max_ratio', 0.2), p=p))
        elif t_type == "VolumeScale":
            pipeline.append(VolumeScale(gain_min=cfg.get('gain_min', 0.5), gain_max=cfg.get('gain_max', 1.5), p=p))
        elif t_type == "RIR_Simulation":
            pipeline.append(RIRSimulation(rir_path=cfg.get('rir_path', None), p=p))
        elif t_type == "DynamicSNRMixing":
            pipeline.append(DynamicSNRMixing(noise_dataset_path=cfg.get('noise_dataset_path', None), p=p))
            
    return torch.nn.Sequential(*pipeline)

def build_freq_domain_transforms(transforms_cfg: list) -> torch.nn.Module:
    """根据配置字典实例化频域变换数组为整体流水线 Module。"""
    if not transforms_cfg:
        return torch.nn.Identity()

    pipeline = []
    for cfg in transforms_cfg:
        t_type = cfg.get('type')
        p = cfg.get('p', 0.5)
        
        if t_type == "SpecMasking":
            pipeline.append(SpecMasking(
                time_mask_param=cfg.get('time_mask_param', 30), 
                freq_mask_param=cfg.get('freq_mask_param', 15), 
                p=p
            ))
        elif t_type == "FilterAugment":
            pipeline.append(FilterAugment(
                n_band=cfg.get('n_band', 1),
                db_range=tuple(cfg.get('db_range', [-5.0, 5.0])),
                band_width_ratio=cfg.get('band_width_ratio', 0.2),
                p=p
            ))
        elif t_type == "VTLP":
            pipeline.append(VTLP(
                warp_factor_range=tuple(cfg.get('warp_factor_range', [0.9, 1.1])),
                p=p
            ))
        elif t_type == "SpecMix":
            pipeline.append(SpecMix(p=p))
            
    return torch.nn.Sequential(*pipeline)
