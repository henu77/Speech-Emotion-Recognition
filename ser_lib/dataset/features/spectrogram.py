import torch
import torch.nn as nn
import torchaudio.transforms as T

class SpectrogramExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = T.Spectrogram(**kwargs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)

class MelSpectrogramExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = T.MelSpectrogram(**kwargs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)

class LogMelSpectrogramExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(**kwargs)
        self.log_transform = T.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.log_transform(mel_spec)
        return log_mel_spec

class MFCCExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = T.MFCC(**kwargs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)

def build_spectrogram_extractor(spec_type: str, spec_kwargs: dict, target_sr: int) -> nn.Module:
    """构建单输出谱图提取器"""
    if spec_type == 'Spectrogram':
        return SpectrogramExtractor(**spec_kwargs)
    elif spec_type == 'MelSpectrogram':
        if 'sample_rate' not in spec_kwargs:
            spec_kwargs['sample_rate'] = target_sr
        return MelSpectrogramExtractor(**spec_kwargs)
    elif spec_type == 'LogMelSpectrogram':
        if 'sample_rate' not in spec_kwargs:
            spec_kwargs['sample_rate'] = target_sr
        return LogMelSpectrogramExtractor(**spec_kwargs)
    elif spec_type == 'MFCC':
        if 'sample_rate' not in spec_kwargs:
            spec_kwargs['sample_rate'] = target_sr
        return MFCCExtractor(**spec_kwargs)
    else:
        raise ValueError(f"不支持的谱图类型: {spec_type}")
