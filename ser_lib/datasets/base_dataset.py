import json
import yaml
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional

from ser_lib.datasets.augment.time_domain import build_time_domain_transforms
from ser_lib.datasets.augment.freq_domain import build_freq_domain_transforms

class BaseConfigDataset(Dataset):
    """配置驱动的语音情感识别 (SER) 数据集核心引擎。
    
    该类负责单样本的生命周期管理，支持懒加载长音频片段、多模态特征并行提取，
    以及基于策略配置的时域与频域双重数据增强流水线。
    
    Attributes:
        split (str): 数据集划分 ('train', 'val', 'test')。
        config (dict): 解析后的 YAML 配置字典。
        target_sr (int): 全局统一的目标音频采样率。
        data_list (List[dict]): 包含音频路径及标签等元数据的列表。
        id2label (dict): 标签 ID 到名称的映射，格式为 {0: {"en": "Neutral", "zh": "平静"}, ...}。
    """
    
    def __init__(self, config_path: str, split: str = 'train'):
        """初始化数据集。
        
        Args:
            config_path (str): YAML 配置文件路径。
            split (str): 数据集模式，默认为 'train'。
        """
        super().__init__()
        self.split = split
        self.config = self._load_config(config_path)
        
        # 1. 解析基础配置与路径映射
        self.target_sr = self.config.get('audio', {}).get('target_sr', 16000)
        list_file = self.config['data_lists'][split]
        self.metadata_dir = self.config.get('paths', {}).get('metadata_dir', None)
        
        if self.metadata_dir and not Path(list_file).is_absolute():
            if not str(list_file).startswith(self.metadata_dir):
                list_file = str(Path(self.metadata_dir) / list_file)
        
        self.data_list = self._load_data_list(list_file)
        self.data_root_dir = self.config.get('paths', {}).get('data_root_dir', None)
        
        # 提取标签映射关系供推理和可视化使用
        self.id2label = self.config.get('class_mapping', {})
        
        # 2. 构建数据增强流水线 (Transforms Pipeline)
        transforms_cfg = self.config.get('transforms', {})
        
        # 2.1 基础波形增强 (课程2)
        self.wave_transform = build_time_domain_transforms(
            transforms_cfg.get('waveform_level', {}).get(split, []), 
            sample_rate=self.target_sr
        )
        # 2.2 高级波形增强 (课程7: 解决工程数据偏差预留)
        self.adv_wave_transform = build_time_domain_transforms(
            transforms_cfg.get('advanced_waveform_level', {}).get(split, []), 
            sample_rate=self.target_sr
        )
        # 2.3 频谱级增强 (课程3/4)
        self.spec_transform = build_freq_domain_transforms(
            transforms_cfg.get('spectrogram_level', {}).get(split, [])
        )

        # 3. 动态构建多模态声学特征提取器字典
        self.feature_extractors = self._build_feature_extractors()

    def _load_config(self, config_path: str) -> dict:
        """加载并解析 YAML 配置文件。"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_data_list(self, list_path: str) -> List[dict]:
        """加载 JSON 或 JSONL 格式的数据集元数据列表。"""
        path = Path(list_path)
        if not path.exists():
            raise FileNotFoundError(f"数据列表文件不存在: {list_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            elif path.suffix == '.jsonl':
                return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                raise ValueError(f"不支持的数据格式: {path.suffix}. 仅支持 .json 或 .jsonl")

    def _load_config(self, config_path: str) -> dict:

    def _build_feature_extractors(self) -> torch.nn.ModuleDict:
        """根据配置字典，实例化对应的 torchaudio 特征提取模块。"""
        extractors = torch.nn.ModuleDict()
        features_cfg = self.config.get('features', {})
        
        for feat_name, cfg in features_cfg.items():
            feat_type = cfg.get('type')
            
            # 反射解析窗函数对象
            window_fn = torch.hann_window if cfg.get('window_fn') == 'hann_window' else None
            
            if feat_type == "Spectrogram":
                extractors[feat_name] = T.Spectrogram(
                    n_fft=cfg.get('n_fft', 400), hop_length=cfg.get('hop_length'), 
                    power=cfg.get('power', 2.0), window_fn=window_fn
                )
            elif feat_type == "LogMelSpectrogram":
                stype = 'power' if cfg.get('power', 2.0) == 2.0 else 'magnitude'
                extractors[feat_name] = torch.nn.Sequential(
                    T.MelSpectrogram(
                        sample_rate=cfg.get('sample_rate', self.target_sr),
                        n_fft=cfg.get('n_fft', 400), hop_length=cfg.get('hop_length'), 
                        n_mels=cfg.get('n_mels', 80), power=cfg.get('power', 2.0), 
                        window_fn=window_fn
                    ),
                    T.AmplitudeToDB(stype=stype, top_db=cfg.get('top_db', 80.0))
                )
            elif feat_type == "MFCC":
                melkwargs = cfg.get('melkwargs', {})
                if melkwargs.get('window_fn') == 'hann_window': 
                    melkwargs['window_fn'] = torch.hann_window
                extractors[feat_name] = T.MFCC(
                    sample_rate=cfg.get('sample_rate', self.target_sr), 
                    n_mfcc=cfg.get('n_mfcc', 40), dct_type=cfg.get('dct_type', 2), 
                    log_mels=cfg.get('log_mels', True), melkwargs=melkwargs
                )
            elif feat_type == "Raw":
                extractors[feat_name] = torch.nn.Identity()
                
        return extractors

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        """获取单个样本的特征、标签与有效长度。
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, int]: 
                - 包含各类提取特征的字典 (如 {"log_mel": tensor})
                - 情感类别标签
                - 特征在时间轴上的实际长度 (用于后续的掩码对齐)
        """
        item = self.data_list[idx]
        audio_path = str(item['audio_path'])
        
        # 拼接绝对路径
        if not Path(audio_path).is_absolute() and self.data_root_dir:
            audio_path = str(Path(self.data_root_dir) / audio_path)
            
        label = item['label']
        
        # 1. 懒加载波形：若配置了时间范围，则利用 C++ 指针偏移仅读取指定片段
        start_ms, end_ms = item.get('start_time_ms', 0), item.get('end_time_ms', None)
        if start_ms > 0 or end_ms is not None:
            orig_sr = torchaudio.info(audio_path).sample_rate
            frame_offset = int((start_ms / 1000.0) * orig_sr)
            num_frames = int(((end_ms - start_ms) / 1000.0) * orig_sr) if end_ms else -1 
            waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        else:
            waveform, sr = torchaudio.load(audio_path)
        
        # 强制单声道与重采样
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr: 
            waveform = T.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        # 2. 波形级数据增强 (基础增强 + 课程7高级鲁棒增强)
        if self.wave_transform:
            waveform = self.wave_transform(waveform)
        if self.adv_wave_transform:
            waveform = self.adv_wave_transform(waveform)
            
        # 3. 动态特征提取与频谱级增强 (SpecAugment)
        extracted_features = {}
        if len(self.feature_extractors) == 0:
            extracted_features['raw_waveform'] = waveform.squeeze(0)
        else:
            for feat_name, extractor in self.feature_extractors.items():
                feat = extractor(waveform)
                
                # 仅当特征为 2D 且配置了 SpecAugment 时执行掩码操作
                if not isinstance(extractor, torch.nn.Identity) and self.spec_transform:
                    feat = self.spec_transform(feat)
                    
                extracted_features[feat_name] = feat.squeeze(0) 

        # 4. 获取序列有效长度 (以字典中首个特征的时间维度为准)
        seq_length = list(extracted_features.values())[0].shape[-1]
            
        return extracted_features, torch.tensor(label, dtype=torch.long), seq_length

    # --- 辅助方法区 ---
    def get_labels(self) -> List[int]:
        """返回所有标签集合，用于计算类别权重或做平衡采样。"""
        return [item['label'] for item in self.data_list]


