import json
import yaml
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple


class BaseConfigDataset(Dataset):
    """
    完全由 YAML 配置驱动的语音情感识别数据集基类。
    所有数据集特定的逻辑（类别映射、路径、增强）均在 {dataset_name}.yaml 中定义。
    列表文件规定格式为包含字典的 JSON 或 JSONL (如 `{"audio_path": "...", "label": 1, "speaker_id": "01"}`)。
    """
    
    def __init__(self, config_path: str, split: str = 'train'):
        super().__init__()
        self.split = split
        self.config = self._load_config(config_path)
        
        # 1. 提取基础参数
        self.target_sr = self.config.get('audio', {}).get('target_sr', 16000)
        self.max_duration = self.config.get('audio', {}).get('max_duration', None)
        
        # 2. 读取数据列表
        list_file = self.config['data_lists'][split]
        self.data_list = self._load_data_list(list_file)
        
        # 3. 初始化增强 (这里可以根据业务具体对接真正的 Transform, 此处保留占位与简单示例)
        self.transforms_cfg = self.config.get('transforms', {}).get(split, [])
        self.transform = self._build_transforms(self.transforms_cfg)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_data_list(self, list_path: str) -> List[dict]:
        path = Path(list_path)
        if not path.exists():
            raise FileNotFoundError(f"Data list file not found: {list_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            elif path.suffix == '.jsonl':
                return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported data list format: {path.suffix}. Use .json or .jsonl")

    def _build_transforms(self, transforms_cfg: list):
        # 示例：根据配置文件动态构建 torchvision/torchaudio transforms 管道
        # 实际项目中可以映射到真实的类：如 {"AddNoise": MyAddNoiseTransform}
        def apply_transforms(waveform):
            # 将基础变换作为闭包或 nn.Sequential 返回
            # 这里是占位符，可根据实际支持的 transform 名称库扩展
            for t_cfg in transforms_cfg:
                t_type = t_cfg.get('type')
                if t_type == "Normalize":
                    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
                # elif t_type == "AddNoise": ...
            return waveform
        return apply_transforms

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        audio_path = item['audio_path']
        label = item['label']
        
        # 读取此音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 强制单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 强制指定采样率
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 截断或填充到指定时长
        if self.max_duration is not None:
            max_len = int(self.max_duration * self.target_sr)
            curr_len = waveform.shape[1]
            if curr_len > max_len:
                # 简单截断
                waveform = waveform[:, :max_len]
            elif curr_len < max_len:
                # 0 填充
                pad_len = max_len - curr_len
                waveform = F.pad(waveform, (0, pad_len), "constant", 0)

        # 运用数据增强
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, torch.tensor(label, dtype=torch.long)

    # ------------------ 辅助与工程方法 ------------------

    def get_labels(self) -> List[int]:
        """返回数据集中所有的标签，可用于计算 Class Weights 或做平衡采样"""
        return [item['label'] for item in self.data_list]

    def get_class_names(self) -> Dict[int, str]:
        """返回 ID_TO_NAME 映射"""
        return self.config.get('class_mapping', {})
    
    def get_speaker_ids(self) -> List[str]:
        """返回所有的 speaker_id (如果有这一列)，用于 K-Fold 分折"""
        return [str(item.get('speaker_id', 'unknown')) for item in self.data_list]
