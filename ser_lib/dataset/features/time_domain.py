import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ==============================================================================
# 1. 核心时域韵律特征 (Time-Domain Prosodic Features)
# ==============================================================================

class PitchF0(nn.Module):
    """提取基频 F0 (Pitch)特征，反映发音人的音高（如愤怒时基频升高）。"""
    def __init__(self, sample_rate: int = 16000, hop_length: int = 256):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [Batch, Time] or [Time]
        # torchaudio.functional.detect_pitch_frequency 基于归一化互相关(NCCF)算法
        # 返回: [Batch, Freq, Time] 但是由于它专门算基频，Freq维度通常是1
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform, 
            sample_rate=self.sample_rate, 
            frame_time=self.hop_length / self.sample_rate,
            win_length=int(self.sample_rate * 0.03), # 30ms window
            freq_low=50, 
            freq_high=800
        )
        return pitch


class RMS(nn.Module):
    """短时能量/均方根 (Root Mean Square) 特征，反映声音的响度/能量。"""
    def __init__(self, win_length: int = 400, hop_length: int = 256):
        super().__init__()
        # 使用 1D AvgPool 来模拟移动窗口求平方和平均
        self.win_length = win_length
        self.hop_length = hop_length
        self.pool = nn.AvgPool1d(kernel_size=win_length, stride=hop_length, padding=win_length//2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T] -> [B, 1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        x = waveform.unsqueeze(1)
        
        # 能量 = sqrt( Mean(x^2) )
        x_sq = x ** 2
        rms = torch.sqrt(self.pool(x_sq) + 1e-8)
        
        # [B, 1, T_frames] 面向下游特征通常去掉通道维度
        return rms.squeeze(1)


class ZeroCrossingRate(nn.Module):
    """过零率 (ZCR) 特征，高频成分多的声音(如清音/摩擦音)过零率高。"""
    def __init__(self, win_length: int = 400, hop_length: int = 256):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.pool = nn.AvgPool1d(kernel_size=win_length, stride=hop_length, padding=win_length//2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # ZCR = 0.5 * sum( |sgn(x[n]) - sgn(x[n-1])| ) / win_length
        # 使用 torch.sign
        signs = torch.sign(waveform)
        # 求一阶差分
        diffs = torch.abs(signs[:, 1:] - signs[:, :-1])
        # 为了保持长度，差分左侧补一个0
        diffs = F.pad(diffs, (1, 0))
        
        diffs = diffs.unsqueeze(1) # [B, 1, T]
        # 通过均值池化，直接相当于滑动窗口求和并除以窗长
        zcr = 0.5 * self.pool(diffs)
        return zcr.squeeze(1)

# ==============================================================================
# 2. 时域音质特征 (Time-Domain Voice Quality) 针对微观波动提取
# ==============================================================================

class JitterShimmerHNR(nn.Module):
    """联合提取音质波动特征 (Jitter/Shimmer/HNR)。
       由于这三个特征紧密依赖声带高精基频 T0 周期，此处利用差分序列聚合模拟。
       返回的是 utterance(音频片段) 级别的标量特征。
    """
    def __init__(self):
        super().__init__()
        # 在工程实现中，通常使用类似 Praat 的独立组件提取。
        # 纯 PyTorch 近似实现较为繁琐，这部分建议使用 torch.diff 对时域波形峰谷或 Pitch 序列评估。
        pass
    
    def forward(self, waveform: torch.Tensor, pitch_f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, Samples]
            pitch_f0: [B, Time] 由 PitchF0 提取出来的帧级基频序列
        Returns:
            [B, 3] 分别对应 Jitter, Shimmer, HNR 这三个标量值。
        """
        B = waveform.shape[0]
        # 【局部相对 Jitter 模拟】：相邻 F0 差值 / 母体 F0 均值
        T0_frames = 1.0 / (pitch_f0 + 1e-5) # 周期矩阵
        # 仅仅处理有基频存在的有话段 (F0 > 0)
        valid_mask = pitch_f0 > 0
        
        jitters = []
        for b in range(B):
            valid_T0 = T0_frames[b][valid_mask[b]]
            if len(valid_T0) < 2:
                jitters.append(0.0)
            else:
                diffs = torch.abs(valid_T0[1:] - valid_T0[:-1])
                jitter = torch.mean(diffs) / (torch.mean(valid_T0) + 1e-8)
                jitters.append(jitter.item())
        jitter_tensor = torch.tensor(jitters, device=waveform.device)
        
        # 为了极简，我们将这 3 种音质特征作为一个联合标量张量抛出供后续 MLP 消化
        # 注意: Shimmer 和精确 HNR 的计算严谨做法需使用 PRAAT 内核算法，
        # 在端到端深度学习中由于其过于脆弱，不常用于主路。这里留桩填 0。
        shimmer_tensor = torch.zeros_like(jitter_tensor)
        hnr_tensor = torch.zeros_like(jitter_tensor)
        
        return torch.stack([jitter_tensor, shimmer_tensor, hnr_tensor], dim=-1) # [B, 3]
