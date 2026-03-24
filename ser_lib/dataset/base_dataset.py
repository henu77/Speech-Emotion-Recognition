import json
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ser_lib.dataset.config_schema import load_config


class BaseConfigDataset(Dataset):
    """配置驱动的 SER 数据集基类。

    仅负责公共的生命周期管理：
      - 解析 YAML 配置
      - 加载 JSON/JSONL 数据列表
      - 懒加载音频并完成单声道转换与重采样

    子类需覆写 ``_load_item(waveform, item)`` 来实现各自的特征提取逻辑，
    以及可选地覆写 ``__getitem__`` 来控制返回格式。

    Attributes:
        split (str): 数据集划分 ('train', 'val', 'test')。
        config (dict): 解析后的 YAML 配置字典。
        target_sr (int): 全局统一的目标音频采样率。
        data_list (List[dict]): 包含音频路径及标签等元数据的列表。
        id2label (dict): 标签 ID 到名称的映射。
    """

    def __init__(self, config_path: str, split: str = 'train'):
        """
        Args:
            config_path (str): YAML 配置文件路径。
            split (str): 数据集模式，默认 'train'。
        """
        super().__init__()
        self.split = split
        self.config = self._load_config(config_path)
        self.config_dict = self.config.model_dump(mode='python')

        # --- 基础配置解析 ---
        self.target_sr = self.config_dict.get('audio', {}).get('target_sr', 16000)
        self.id2label = self.config_dict.get('class_mapping', {})

        # --- 数据列表路径拼接 ---
        list_file = self.config_dict['data_lists'][split]
        metadata_dir = self.config_dict.get('paths', {}).get('metadata_dir', None)
        if metadata_dir and not Path(list_file).is_absolute():
            if not str(list_file).startswith(metadata_dir):
                list_file = str(Path(metadata_dir) / list_file)

        self.data_list = self._load_data_list(list_file)
        self.data_root_dir = self.config_dict.get('paths', {}).get('data_root_dir', None)
        self.resamplers: Dict[int, torch.nn.Module] = {}

    # ------------------------------------------------------------------
    # 公共辅助方法
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> Any:
        """加载、校验并返回结构化配置对象。"""
        return load_config(config_path)

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
                raise ValueError(f"不支持的数据格式: {path.suffix}，仅支持 .json 或 .jsonl")

    def _load_waveform(self, item: dict) -> torch.Tensor:
        """懒加载音频波形，自动完成单声道转换与重采样。

        支持通过 item 中的 ``start_time_ms`` / ``end_time_ms`` 实现片段级 lazy-load，
        避免加载整段长音频带来的 IO 开销。

        Args:
            item (dict): 数据列表中的单条元数据记录。

        Returns:
            torch.Tensor: shape ``[1, T]`` 的单声道波形，采样率为 ``self.target_sr``。
        """
        audio_path = str(item['audio_path'])
        if not Path(audio_path).is_absolute() and self.data_root_dir:
            audio_path = str(Path(self.data_root_dir) / audio_path)

        start_ms = item.get('start_time_ms', 0)
        end_ms = item.get('end_time_ms', None)

        if start_ms > 0 or end_ms is not None:
            orig_sr = item.get('sample_rate') or item.get('sr')
            if not orig_sr:
                orig_sr = torchaudio.info(audio_path).sample_rate
            frame_offset = int((start_ms / 1000.0) * orig_sr)
            num_frames = int(((end_ms - start_ms) / 1000.0) * orig_sr) if end_ms else -1
            waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        else:
            waveform, sr = torchaudio.load(audio_path)

        # 强制单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # 重采样到目标采样率
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self.resamplers[sr](waveform)

        return waveform  # [1, T]

    # ------------------------------------------------------------------
    # Dataset 协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        """子类必须覆写此方法，实现特征提取并返回特征字典与序列长度。

        Args:
            waveform (torch.Tensor): shape ``[1, T]`` 的原始波形。
            item (dict): 当前样本的元数据记录。

        Returns:
            Tuple[Dict[str, torch.Tensor], int]:
                - features: 特征字典，key 为特征名，value 为 Tensor。
                - seq_length: 特征在时间轴上的有效帧数。
        """
        raise NotImplementedError("子类必须实现 _load_item 方法")

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        """获取单个样本。

        Returns:
            Tuple:
                - features (Dict[str, Tensor]): 特征字典。
                - label (Tensor): 情感类别标签，dtype=torch.long。
                - seq_length (int): 特征时间轴有效帧数，供 collate_fn 使用。
        """
        item = self.data_list[idx]
        label = item['label']
        waveform = self._load_waveform(item)

        features, seq_length = self._load_item(waveform, item)

        return features, torch.tensor(label, dtype=torch.long), seq_length

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_labels(self) -> List[int]:
        """返回全部标签列表，用于计算类别权重或平衡采样。"""
        return [item['label'] for item in self.data_list]
