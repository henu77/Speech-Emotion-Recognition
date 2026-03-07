import torch
from typing import Dict, Tuple

from ser_lib.dataset.base_dataset import BaseConfigDataset


class CustomDataset(BaseConfigDataset):
    """用户自定义特征数据集（预留扩展口）。

    任务定位: 供用户自由实现任意复杂的特征提取逻辑，例如：
      - 调用预训练大模型（Wav2Vec2、HuBERT）提取 embedding
      - 读取外部预提取的特征文件（.npy / .pt）
      - 组合多种特征并做自定义融合

    配置文件: custom_dataset.yaml

    使用方式:
        继承本类并覆写 ``_load_item``，在 ``__init__`` 中初始化所需模型/提取器。

    示例 (Wav2Vec2 embedding)::

        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

        class Wav2Vec2Dataset(CustomDataset):
            def __init__(self, config_path, split='train'):
                super().__init__(config_path, split)
                model_name = "facebook/wav2vec2-base"
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = Wav2Vec2Model.from_pretrained(model_name).eval()

            def _load_item(self, waveform, item):
                audio_np = waveform.squeeze(0).numpy()
                inputs = self.processor(audio_np, sampling_rate=self.target_sr, return_tensors="pt")
                with torch.no_grad():
                    hidden = self.model(**inputs).last_hidden_state.squeeze(0).T  # [D, T]
                return {"wav2vec2": hidden}, hidden.shape[-1]
    """

    def __init__(self, config_path: str, split: str = 'train'):
        super().__init__(config_path, split)
        self._custom_cfg = self.config.get('features', {}).get('custom', {})
        # TODO: 初始化你的自定义模型或特征提取器

    def _load_item(self, waveform: torch.Tensor, item: dict) -> Tuple[Dict[str, torch.Tensor], int]:
        """用户自定义特征提取（请根据需求实现）。

        Args:
            waveform: ``[1, T]`` 原始波形，采样率为 ``self.target_sr``。
            item: 样本元数据，可读取 audio_path、label 及其他自定义字段。
        """
        features: Dict[str, torch.Tensor] = {}
        # TODO: 实现自定义特征提取
        # features["my_feat"] = self.model(waveform).squeeze(0)
        seq_length = list(features.values())[0].shape[-1] if features else 0
        return features, seq_length
