import torch
import random
import torchaudio.transforms as T
import torch.nn.functional as F

class SpecMasking(torch.nn.Module):
    """SpecAugment 的核心：时间掩码与频率掩码。"""
    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 15, p: float = 0.5):
        super().__init__()
        self.p = p
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # SpecAugment 分别评估时域与频域的掩码触发概率
        if random.random() < self.p:
            spec = self.time_masking(spec)
        if random.random() < self.p:
            spec = self.freq_masking(spec)
        return spec


class FilterAugment(torch.nn.Module):
    """滤波器增强 (随机对特定频带的能量进行放缩)。
       由于主流输入是 Log-Mel 谱，乘法增益在此处体现为数学加法。"""
    def __init__(self, n_band: int = 1, db_range: tuple = (-5.0, 5.0), band_width_ratio: float = 0.2, p: float = 0.5):
        super().__init__()
        self.p = p
        self.n_band = n_band
        self.db_range = db_range
        self.band_width_ratio = band_width_ratio

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return spec
            
        freq_bins = spec.shape[-2]
        band_width = max(1, int(freq_bins * self.band_width_ratio))
        
        # 初始化 0 增益掩码向量
        add_weight = torch.zeros(freq_bins, device=spec.device)
        
        for _ in range(self.n_band):
            # 随机起点
            start = random.randint(0, freq_bins - band_width)
            db_change = random.uniform(*self.db_range)
            add_weight[start:start+band_width] += db_change
            
        # 扩展时间轴并相加 (Log-mel 域中的加法相当于线性域的乘法滤波器)
        spec = spec + add_weight.unsqueeze(-1)
        return spec


class VTLP(torch.nn.Module):
    """声道长度扰动 (Vocal Tract Length Perturbation)。
       通过对频率轴(Y轴)进行物理拉伸和截断，极其有效地模拟不用性别/年龄带来的声带共振差异。"""
    def __init__(self, warp_factor_range: tuple = (0.9, 1.1), p: float = 0.5):
        super().__init__()
        self.p = p
        self.warp_factor_range = warp_factor_range

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return spec
            
        warp_factor = random.uniform(*self.warp_factor_range)
        if warp_factor == 1.0:
            return spec
            
        # spec shape 可能是 [Freq, Time]
        orig_freq = spec.shape[-2]
        
        # 将 Freq 轴转到 Time 的位置使用 1D Interpolate，
        # 组装为 [Batch(Time), Channel(1), Temporal(Freq)]
        spec_t = spec.transpose(-1, -2).unsqueeze(1) 
        
        new_freq = int(orig_freq * warp_factor)
        warped_t = F.interpolate(spec_t, size=new_freq, mode='linear', align_corners=False)
        
        # 旋转回来: [Freq_warped, Time]
        warped = warped_t.squeeze(1).transpose(-1, -2)
        
        # 为保证与模型输入尺寸对齐，进行顶部截断或底噪补齐
        if new_freq > orig_freq:
            # 切除超出的高频部分
            warped = warped[..., :orig_freq, :]
        else:
            # 频率被压缩后，高频出现空缺，用全图最小背景值补齐
            pad_len = orig_freq - new_freq
            pad_val = spec.min().item()
            warped = F.pad(warped, (0, 0, 0, pad_len), value=pad_val)
            
        return warped


class SpecMix(torch.nn.Module):
    """CutMix / SpecMix 的图像矩阵截断拼接算法。"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return spec
        # 单样本管道无法凭空变出另一个样本来叠加，必须依靠 Dataloader 外部支持。
        raise NotImplementedError(
            "【进阶错误】SpecMix / CutMix 是强依赖第二张图纸与双重交叉标签的高级增强手段！ "
            "它必须在 batch-level (拼批之后) 发生。请移动至 collate_fn 中的第 3 级增强阶段实现！"
        )


