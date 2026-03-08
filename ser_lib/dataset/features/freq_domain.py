import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ==============================================================================
# 1. 频谱系特征 (Spectral Features)
# ==============================================================================



class SpectralCentroid(nn.Module):
    """谱质心：表示频谱的“重心”频率位置，衡量声音的明亮程度。"""
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 计算 STFT 使用的参数
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        # 频率轴(Hz): [n_fft//2 + 1] -> [Freq_bins, 1] 方便后面矩阵广播乘法
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1).unsqueeze(-1)
        self.register_buffer("freqs", freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. 提取线性频域幅度谱 S
        S = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window,
            return_complex=True
        ).abs() # [B, Freq, Time]
        
        # 2. 谱质心 = sum(f * S) / sum(S)
        # self.freqs: [Freq, 1], 广播到 [B, Freq, Time]
        numerator = torch.sum(S * self.freqs, dim=-2)
        denominator = torch.sum(S, dim=-2) + 1e-8
        centroid = numerator / denominator
        
        return centroid # [B, Time]


class SpectralRolloff(nn.Module):
    """谱滚降：累积幅度达到总能量特定比例（如 85%）的频率点。区分谐波与噪声成分的指标。"""
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256, roll_percent: float = 0.85):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.roll_percent = roll_percent
        
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        self.register_buffer("freqs", freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        S = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window,
            return_complex=True
        ).abs() # [B, Freq, Time]
        
        # 计算每一帧的总能量 E_total [B, Time]
        total_energy = torch.sum(S, dim=-2, keepdim=True)
        threshold = total_energy * self.roll_percent
        
        # 沿频率轴累积求和 [B, Freq, Time]
        cumsum_S = torch.cumsum(S, dim=-2)
        
        # 找到刚刚超过阈值的那个频率索引 (cumsum > threshold)
        # mask = cumsum_S >= threshold 得到 True/False 矩阵
        # 利用 argmax 找到第一个 True 的索引位置
        mask = cumsum_S >= threshold
        
        # [B, Time]
        rolloff_idx = torch.argmax(mask.to(torch.int8), dim=-2)
        
        # 映射回 Hz
        rolloff_hz = self.freqs[rolloff_idx]
        return rolloff_hz


class SpectralFlatness(nn.Module):
    """谱平坦度：几何平均值与算术平均值的比值。(0->纯音，1->白噪音)"""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.eps = 1e-10

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        S = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window,
            return_complex=True
        ).abs() # [B, Freq, Time]
        
        # Power spectrogram 频谱能量
        S_power = S ** 2 + self.eps
        # 几何平均值公式推导: exp( mean( log(x) ) )
        gmean = torch.exp(torch.mean(torch.log(S_power), dim=-2))
        amean = torch.mean(S_power, dim=-2)
        
        flatness = gmean / (amean + self.eps)
        return flatness # [B, Time]


class SpectralFlux(nn.Module):
    """谱通量：相邻帧频谱之间变化速率。起音(Onset)时很大。"""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        S = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window,
            return_complex=True
        ).abs() # [B, Freq, Time]
        
        # HWR (Half-Wave Rectified): max(0, S_t - S_{t-1})
        diff = S[..., 1:] - S[..., :-1]
        diff = F.relu(diff)
        
        flux = torch.sum(diff ** 2, dim=-2) # 欧几里德距离平方和 [B, Time-1]
        
        # 补齐长度使得 Time 轴与原来一样长 (首帧通量设为0)
        flux = F.pad(flux, (1, 0))
        return flux

# ==============================================================================
# 2. 差分动态特征辅助函数 (Delta 和 Delta-Delta) -> 通常作用于频域矩阵
# ==============================================================================

class ComputeDeltas(nn.Module):
    """计算任意特征矩阵的差分动态特征。"""
    def __init__(self, win_length: int = 5):
        super().__init__()
        assert win_length >= 3 and win_length % 2 == 1, "窗长必须为>=3的奇数"
        self.transform = T.ComputeDeltas(win_length=win_length)
        
    def forward(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        # 要求特征矩阵至少是 3D，例如 MFCC: [B, N_freq, Time]
        if feature_matrix.dim() == 2:
            # 强行扩展出统一维度 [B, 1, Time]
            feature_matrix = feature_matrix.unsqueeze(1)
            
        delta = self.transform(feature_matrix)
        delta_delta = self.transform(delta)
        
        # 将原始特征、一阶导数、二阶导数沿“通道/频率”维度拼接
        # [B, C*3, Time]
        return torch.cat([feature_matrix, delta, delta_delta], dim=-2)
