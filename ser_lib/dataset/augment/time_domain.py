import torch
import random
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

class AddGaussianNoise(torch.nn.Module):
    """向波形注入高斯白噪声。"""
    def __init__(self, snr: float = 15.0, p: float = 0.5):
        super().__init__()
        self.snr = snr
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        
        noise = torch.randn_like(waveform)
        # 计算信号和噪声的有效功率 (均方值)
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # 避免全零音频出现的无穷大
        if signal_power == 0:
            return waveform
            
        # 根据目标信噪比 (SNR = 10 * log10(Ps/Pn)) 计算目标噪声功率
        target_noise_power = signal_power / (10 ** (self.snr / 10))
        # 求出缩放系数
        scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))
        
        return waveform + scale * noise


class PitchShift(torch.nn.Module):
    """音频音高偏移 (使用 torchaudio 官方支持的特征变换，若无则使用重采样模拟)。"""
    def __init__(self, sample_rate: int, n_steps: int = 4, p: float = 0.5):
        super().__init__()
        self.p = p
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        
        try:
            # Pytorch 2.1+ 支持直接使用 torchaudio.transforms.PitchShift
            self.transform = T.PitchShift(sample_rate, n_steps)
            self.use_resample = False
        except AttributeError:
            self.use_resample = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
            
        if not self.use_resample:
            return self.transform(waveform)
            
        # 降级：若低版本无 PitchShift，通过改变采样率后恢复原采样率来实现音调改变（伴随语速改变）
        factor = 2 ** (self.n_steps / 12)
        new_freq = int(self.sample_rate * factor)
        shifted = torchaudio.functional.resample(waveform, self.sample_rate, new_freq)
        return torchaudio.functional.resample(shifted, new_freq, self.sample_rate)


class TimeStretch(torch.nn.Module):
    """音频时间拉伸 (改变语速不改变音调)。"""
    def __init__(self, rate: float = 1.2, p: float = 0.5):
        super().__init__()
        self.p = p
        self.rate = rate
        # True Phase Vocoder 必须在 STFT 频域进行，为了保持纯 1D 时域接口的简便性，
        # 此处使用 Torchaudio 的相位声码器配合内部傅里叶变换代理封装：
        self.n_fft = 1024
        self.hop_length = 256
        self.stretch = T.TimeStretch(n_freq=(self.n_fft // 2) + 1, hop_length=self.hop_length, fixed_rate=rate)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
            
        # 必须先转化到复数频域，再伸缩，最后逆变换回波形
        window = torch.hann_window(self.n_fft).to(waveform.device)
        stft_complex = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
        )
        
        # 剥离复数为实数与虚数组装，供给 TimeStretch使用
        stft_stretched = self.stretch(stft_complex)
        
        # 还原回时域
        stretched_waveform = torch.istft(
            stft_stretched, n_fft=self.n_fft, hop_length=self.hop_length, window=window
        )
        return stretched_waveform


class TimeShift(torch.nn.Module):
    """时间平移。随机向前或向后平移音频并用零填充。"""
    def __init__(self, shift_max_ratio: float = 0.2, p: float = 0.5):
        super().__init__()
        self.shift_max_ratio = shift_max_ratio
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
            
        max_shift = int(waveform.shape[-1] * self.shift_max_ratio)
        shift_amt = random.randint(-max_shift, max_shift)
        
        if shift_amt == 0:
            return waveform
        elif shift_amt > 0:
            # 向右平移，左边补零，右边截断
            shifted = F.pad(waveform[..., :-shift_amt], (shift_amt, 0), value=0.0)
        else:
            # 向左平移，右侧补零，左侧截断
            shifted = F.pad(waveform[..., -shift_amt:], (0, -shift_amt), value=0.0)
        return shifted


class VolumeScale(torch.nn.Module):
    """音量缩放，模拟麦克风远近振幅波动。"""
    def __init__(self, gain_min: float = 0.5, gain_max: float = 1.5, p: float = 0.5):
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
            
        gain = random.uniform(self.gain_min, self.gain_max)
        return waveform * gain


class RIRSimulation(torch.nn.Module):
    """房间冲激响应(混响)模拟 [高级]。"""
    def __init__(self, rir_path: str = None, p: float = 0.5):
        super().__init__()
        self.p = p
        self.rir_path = rir_path
        # 注意：此处为课程7预留，如果要完全实现，需要读取真实的 RIR 音频并执行 fftconvolve，或提示加载外部库

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        # TODO: 完全实现需要一个纯净的真实的 RIR (Room Impulse Response) 数据库！
        raise NotImplementedError("【课程 7 进阶要求】混响模拟需要真实的房间响应库，请同学们在外部寻找开源 RIR WAV 并结合 torchaudio.functional.fftconvolve 实现！")


class DynamicSNRMixing(torch.nn.Module):
    """动态环境信噪比真实混合 [高级]。"""
    def __init__(self, noise_dataset_path: str = None, p: float = 0.5):
        super().__init__()
        self.p = p
        self.noise_dataset_path = noise_dataset_path

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        raise NotImplementedError("【课程 7 进阶要求】动态真实背景混合需要外部 MUSAN 噪声数据库支撑，请各位自行下载并在此使用文件 IO 加载与音频信噪比例重组！")


