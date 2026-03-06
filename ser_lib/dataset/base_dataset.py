import json
import yaml
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional

from ser_lib.dataset.augment.builder import build_time_domain_transforms
from ser_lib.dataset.loaders.waveform_loader import WaveformLoader
from ser_lib.dataset.loaders.time_domain_loader import TimeDomainFeatureLoader
from ser_lib.dataset.loaders.freq_domain_loader import FreqDomainFeatureLoader
from ser_lib.dataset.loaders.custom_feature_loader import CustomFeatureLoader

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
        # 2.3 实例化各域特征提取加载器
        features_cfg = self.config.get('features', {})
        
        self.waveform_loader = WaveformLoader(
            features_cfg.get('waveform', {})
        )
        self.time_loader = TimeDomainFeatureLoader(
            features_cfg.get('time_domain', {})
        )
        self.freq_loader = FreqDomainFeatureLoader(
            features_cfg.get('freq_domain', {}),
            transforms_cfg.get('spectrogram_level', {}).get(split, [])
        )
        self.custom_loader = CustomFeatureLoader(
            features_cfg.get('custom', {})
        )

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
            
        # 3. 分发给专门的域特征加载器 (由加载器内部处理 2D 频谱增强和维度整理)
        extracted_features = {}
        extracted_features.update(self.waveform_loader(waveform))
        extracted_features.update(self.time_loader(waveform))
        extracted_features.update(self.freq_loader(waveform))
        extracted_features.update(self.custom_loader(waveform))
        
        # 若所有配置均为空，保底返回原始波形
        if not extracted_features:
            extracted_features['raw_waveform'] = waveform.squeeze(0)

        # 4. 获取序列有效长度 (以字典中首个特征的时间维度为准)
        seq_length = list(extracted_features.values())[0].shape[-1]
            
        return extracted_features, torch.tensor(label, dtype=torch.long), seq_length

    # --- 辅助方法区 ---
    def get_labels(self) -> List[int]:
        """返回所有标签集合，用于计算类别权重或做平衡采样。"""
        return [item['label'] for item in self.data_list]


