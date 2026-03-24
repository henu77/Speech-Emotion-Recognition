import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Tuple

from ser_lib.dataset.config_schema import (
    BaseDatasetConfig,
    CollateStrategy,
    DatasetType,
    detect_dataset_type,
    validate_strategy_compatibility,
)


def _to_config_dict(config: Any) -> Dict[str, Any]:
    """将 schema 对象或原始字典统一转换为字典。"""
    if isinstance(config, BaseDatasetConfig):
        return config.model_dump(mode="python")
    if isinstance(config, dict):
        return config
    raise TypeError(f"不支持的配置类型: {type(config)!r}")


def _detect_dataset_type_from_config(config_dict: Dict[str, Any]) -> DatasetType:
    """根据配置字典检测当前数据集类型。"""
    return detect_dataset_type(config_dict)


def _build_base_batch(
    inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lengths: torch.Tensor,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
    mask: torch.Tensor | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """构造统一的 batch 协议。"""
    batch_meta = {
        "dataset_type": dataset_type.value,
        "collate_strategy": strategy.value,
        "window_counts": None,
        "original_labels": None,
    }
    if meta:
        batch_meta.update(meta)

    return {
        "inputs": inputs,
        "labels": labels,
        "lengths": lengths,
        "mask": mask,
        "meta": batch_meta,
    }


def _unpack_batch(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, List[str]]:
    """解包通用 batch 输入。"""
    assert len(batch) > 0, "[collate_fn] batch 为空，无法处理"

    features_list = [item[0] for item in batch]
    labels = torch.stack([
        item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1], dtype=torch.long)
        for item in batch
    ]).long()
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)

    assert labels.shape[0] == lengths.shape[0] == len(batch), (
        f"[collate_fn] 解包维度不一致: batch_size={len(batch)}, labels.shape={labels.shape}, lengths.shape={lengths.shape}"
    )

    feature_names = list(features_list[0].keys())
    return features_list, labels, lengths, feature_names


def _pad_or_truncate(feat: torch.Tensor, target_length: int) -> torch.Tensor:
    """沿最后一维补零或截断到目标长度。"""
    current_length = feat.shape[-1]
    if current_length > target_length:
        return feat[..., :target_length]
    if current_length < target_length:
        return F.pad(feat, (0, target_length - current_length))
    return feat


def _ensure_temporal_feature(feat: torch.Tensor, feat_name: str, context: str) -> torch.Tensor:
    """统一 FeatureDataset 的单特征格式为 [T, D]。"""
    assert feat.dim() in (1, 2), (
        f"[{context}] 特征 '{feat_name}' 维度错误: 期望 1D [T] 或 2D [D, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
    )
    if feat.dim() == 1:
        return feat.unsqueeze(-1)
    return feat.transpose(0, 1)


def _collate_waveform_truncate_pad(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    max_frames: int,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    inputs: Dict[str, torch.Tensor] = {}

    for feat_name in feature_names:
        padded_feats = []
        effective_lengths = []
        for idx, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 1, (
                f"[_collate_waveform_truncate_pad] 样本 {idx} 特征 '{feat_name}' 维度错误: 期望 1D [T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )
            effective_lengths.append(min(feat.shape[-1], max_frames))
            padded_feats.append(_pad_or_truncate(feat, max_frames))
        inputs[feat_name] = torch.stack(padded_feats)
        lengths = torch.tensor(effective_lengths, dtype=torch.long)

    return _build_base_batch(inputs, labels, lengths, dataset_type, strategy)


def _collate_waveform_dynamic_mask(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    max_len = int(lengths.max().item())
    inputs: Dict[str, torch.Tensor] = {}

    for feat_name in feature_names:
        padded_feats = []
        for idx, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 1, (
                f"[_collate_waveform_dynamic_mask] 样本 {idx} 特征 '{feat_name}' 维度错误: 期望 1D [T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )
            padded_feats.append(F.pad(feat, (0, max_len - feat.shape[-1])))
        inputs[feat_name] = torch.stack(padded_feats)

    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return _build_base_batch(inputs, labels, lengths, dataset_type, strategy, mask=mask)


def _collate_waveform_sliding_window(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    window_size: int,
    stride: int,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    inputs: Dict[str, torch.Tensor] = {}
    window_counts: List[int] = []
    expanded_labels: List[int] = []
    expanded_lengths: List[int] = []
    original_labels = labels.clone()

    for feat_name in feature_names:
        all_windows = []
        for i, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 1, (
                f"[_collate_waveform_sliding_window] 样本 {i} 特征 '{feat_name}' 维度错误: 期望 1D [T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )

            current_length = feat.shape[-1]
            sample_windows = []
            sample_lengths = []
            if current_length <= window_size:
                sample_windows.append(F.pad(feat, (0, window_size - current_length)))
                sample_lengths.append(current_length)
            else:
                for start in range(0, current_length - window_size + 1, stride):
                    sample_windows.append(feat[start:start + window_size])
                    sample_lengths.append(window_size)
                if (current_length - window_size) % stride != 0:
                    sample_windows.append(feat[-window_size:])
                    sample_lengths.append(window_size)

            all_windows.extend(sample_windows)
            if feat_name == feature_names[0]:
                num_windows = len(sample_windows)
                window_counts.append(num_windows)
                expanded_labels.extend([labels[i].item()] * num_windows)
                expanded_lengths.extend(sample_lengths)

        inputs[feat_name] = torch.stack(all_windows)

    return _build_base_batch(
        inputs,
        torch.tensor(expanded_labels, dtype=torch.long),
        torch.tensor(expanded_lengths, dtype=torch.long),
        dataset_type,
        strategy,
        meta={
            "window_counts": torch.tensor(window_counts, dtype=torch.long),
            "original_labels": original_labels,
        },
    )


def _collate_spectrogram_truncate_pad(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    max_frames: int,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    inputs: Dict[str, torch.Tensor] = {}
    effective_lengths = torch.clamp(lengths, max=max_frames)

    for feat_name in feature_names:
        padded_feats = []
        for idx, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 2, (
                f"[_collate_spectrogram_truncate_pad] 样本 {idx} 特征 '{feat_name}' 维度错误: 期望 2D [F, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )
            padded_feats.append(_pad_or_truncate(feat, max_frames))
        inputs[feat_name] = torch.stack(padded_feats).unsqueeze(1)

    return _build_base_batch(inputs, labels, effective_lengths, dataset_type, strategy)


def _collate_spectrogram_dynamic_mask(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    max_len = int(lengths.max().item())
    inputs: Dict[str, torch.Tensor] = {}

    for feat_name in feature_names:
        padded_feats = []
        for idx, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 2, (
                f"[_collate_spectrogram_dynamic_mask] 样本 {idx} 特征 '{feat_name}' 维度错误: 期望 2D [F, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )
            padded_feats.append(F.pad(feat, (0, max_len - feat.shape[-1])))
        inputs[feat_name] = torch.stack(padded_feats).transpose(1, 2)

    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return _build_base_batch(inputs, labels, lengths, dataset_type, strategy, mask=mask)


def _collate_spectrogram_sliding_window(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    window_size: int,
    stride: int,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    inputs: Dict[str, torch.Tensor] = {}
    window_counts: List[int] = []
    expanded_labels: List[int] = []
    expanded_lengths: List[int] = []
    original_labels = labels.clone()

    for feat_name in feature_names:
        all_windows = []
        for i, feat_dict in enumerate(features_list):
            feat = feat_dict[feat_name]
            assert feat.dim() == 2, (
                f"[_collate_spectrogram_sliding_window] 样本 {i} 特征 '{feat_name}' 维度错误: 期望 2D [F, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
            )

            current_length = feat.shape[-1]
            sample_windows = []
            sample_lengths = []
            if current_length <= window_size:
                sample_windows.append(F.pad(feat, (0, window_size - current_length)))
                sample_lengths.append(current_length)
            else:
                for start in range(0, current_length - window_size + 1, stride):
                    sample_windows.append(feat[..., start:start + window_size])
                    sample_lengths.append(window_size)
                if (current_length - window_size) % stride != 0:
                    sample_windows.append(feat[..., -window_size:])
                    sample_lengths.append(window_size)

            all_windows.extend(sample_windows)
            if feat_name == feature_names[0]:
                num_windows = len(sample_windows)
                window_counts.append(num_windows)
                expanded_labels.extend([labels[i].item()] * num_windows)
                expanded_lengths.extend(sample_lengths)

        inputs[feat_name] = torch.stack(all_windows).unsqueeze(1)

    return _build_base_batch(
        inputs,
        torch.tensor(expanded_labels, dtype=torch.long),
        torch.tensor(expanded_lengths, dtype=torch.long),
        dataset_type,
        strategy,
        meta={
            "window_counts": torch.tensor(window_counts, dtype=torch.long),
            "original_labels": original_labels,
        },
    )


def _collate_feature_dynamic_mask(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    max_len = int(lengths.max().item())
    inputs: Dict[str, torch.Tensor] = {}

    for feat_name in feature_names:
        padded_feats = []
        for feat_dict in features_list:
            feat = _ensure_temporal_feature(feat_dict[feat_name], feat_name, "_collate_feature_dynamic_mask")
            feat = feat.transpose(0, 1)
            feat = F.pad(feat, (0, max_len - feat.shape[-1]))
            padded_feats.append(feat.transpose(0, 1))
        inputs[feat_name] = torch.stack(padded_feats)

    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return _build_base_batch(inputs, labels, lengths, dataset_type, strategy, mask=mask)


def _collate_feature_truncate_pad(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
    max_frames: int,
    dataset_type: DatasetType,
    strategy: CollateStrategy,
) -> Dict[str, Any]:
    features_list, labels, lengths, feature_names = _unpack_batch(batch)
    inputs: Dict[str, torch.Tensor] = {}
    effective_lengths = torch.clamp(lengths, max=max_frames)

    for feat_name in feature_names:
        padded_feats = []
        for feat_dict in features_list:
            feat = _ensure_temporal_feature(feat_dict[feat_name], feat_name, "_collate_feature_truncate_pad")
            feat = feat.transpose(0, 1)
            feat = _pad_or_truncate(feat, max_frames)
            padded_feats.append(feat.transpose(0, 1))
        inputs[feat_name] = torch.stack(padded_feats)

    return _build_base_batch(inputs, labels, effective_lengths, dataset_type, strategy)


# =====================================================================
# 数据批处理对齐引擎 (Collate Function Factory)
# =====================================================================
def build_collate_fn(config: Dict[str, Any] | BaseDatasetConfig) -> Callable:
    """根据配置动态生成 DataLoader 使用的 collate_fn 批处理闭包。"""
    config_dict = _to_config_dict(config)
    dataset_type = _detect_dataset_type_from_config(config_dict)

    proc_cfg = config_dict.get("audio_processing", {})
    strategy = CollateStrategy(proc_cfg.get("strategy", CollateStrategy.TRUNCATE_PAD.value))
    validate_strategy_compatibility(dataset_type, strategy)

    batch_transforms_cfg = config_dict.get("transforms", {}).get("batch_level", {}).get("train", [])
    enable_mixup = any(t.get("type") == "Mixup" for t in batch_transforms_cfg)

    if dataset_type == DatasetType.WAVEFORM:
        if strategy == CollateStrategy.TRUNCATE_PAD:
            collate_impl = lambda batch: _collate_waveform_truncate_pad(batch, proc_cfg.get("max_frames", 300), dataset_type, strategy)
        elif strategy == CollateStrategy.DYNAMIC_MASK:
            collate_impl = lambda batch: _collate_waveform_dynamic_mask(batch, dataset_type, strategy)
        else:
            collate_impl = lambda batch: _collate_waveform_sliding_window(
                batch,
                proc_cfg.get("window_size", 300),
                proc_cfg.get("stride", 150),
                dataset_type,
                strategy,
            )
    elif dataset_type == DatasetType.SPECTROGRAM:
        if strategy == CollateStrategy.TRUNCATE_PAD:
            collate_impl = lambda batch: _collate_spectrogram_truncate_pad(batch, proc_cfg.get("max_frames", 300), dataset_type, strategy)
        elif strategy == CollateStrategy.DYNAMIC_MASK:
            collate_impl = lambda batch: _collate_spectrogram_dynamic_mask(batch, dataset_type, strategy)
        else:
            collate_impl = lambda batch: _collate_spectrogram_sliding_window(
                batch,
                proc_cfg.get("window_size", 300),
                proc_cfg.get("stride", 150),
                dataset_type,
                strategy,
            )
    else:
        if strategy == CollateStrategy.TRUNCATE_PAD:
            collate_impl = lambda batch: _collate_feature_truncate_pad(batch, proc_cfg.get("max_frames", 300), dataset_type, strategy)
        else:
            collate_impl = lambda batch: _collate_feature_dynamic_mask(batch, dataset_type, strategy)

    def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]) -> Dict[str, Any]:
        final_batch = collate_impl(batch)
        if enable_mixup:
            raise NotImplementedError("请在 collate_fn 中实现 Mixup 批次特征与软标签融合！")
        return final_batch

    return collate_fn
