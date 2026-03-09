import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Callable

# =====================================================================
# 数据批处理对齐引擎 (Collate Function Factory)
# =====================================================================
def build_collate_fn(config: dict) -> Callable:
    """根据配置动态生成 DataLoader 使用的 collate_fn 批处理闭包。
    
    自动处理多特征并发场景下的长短音频补齐，并预留了批处理级别增强的接口。
    
    Args:
        config (dict): 解析后的 YAML 配置字典。
        
    Returns:
        Callable: 供 torch.utils.data.DataLoader 调用的批处理组装函数。
    """
    proc_cfg = config.get('audio_processing', {})
    strategy = proc_cfg.get('strategy', 'truncate_pad')
    
    # 解析【课程 7】批次级增强是否开启 (Mixup)
    batch_transforms_cfg = config.get('transforms', {}).get('batch_level', {}).get('train', [])
    enable_mixup = any(t.get('type') == 'Mixup' for t in batch_transforms_cfg)
    
    def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]) -> Dict[str, Any]:
        """处理单批次数据。"""
        # 维度检查：batch 非空
        assert len(batch) > 0, (
            "[collate_fn] batch 为空，无法处理"
        )

        # 数据解包
        features_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        lengths = torch.tensor([item[2] for item in batch])

        # 维度检查：labels 和 lengths 维度一致
        assert labels.shape[0] == lengths.shape[0] == len(batch), (
            f"[collate_fn] 解包维度不一致: batch_size={len(batch)}, labels.shape={labels.shape}, lengths.shape={lengths.shape}"
        )

        feature_names = features_list[0].keys()
        batch_output = {}

        # -------------------------------------------------------------
        # 阶段 1: 序列时间轴物理对齐 (Padding & Truncating)
        # -------------------------------------------------------------
        if strategy == "truncate_pad":
            max_frames = proc_cfg.get('max_frames', 300)
            for feat_name in feature_names:
                padded_feats = []
                for idx, feat_dict in enumerate(features_list):
                    feat = feat_dict[feat_name]

                    # 维度检查：单样本输入应为 1D [T] 或 2D [Freq, T]
                    assert feat.dim() in (1, 2), (
                        f"[collate_fn.truncate_pad] 样本 {idx} 特征 '{feat_name}' 维度错误: "
                        f"期望 1D [T] 或 2D [Freq, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
                    )

                    current_frames = feat.shape[-1]
                    if current_frames > max_frames:
                        feat = feat[..., :max_frames]
                    elif current_frames < max_frames:
                        feat = F.pad(feat, (0, max_frames - current_frames))
                    padded_feats.append(feat)

                # 为 2D CNN 增加 Channel 维度: 形状变为 [Batch, 1, Freq, Time]
                batch_tensor = torch.stack(padded_feats).unsqueeze(1)

                # 维度检查：batch 输出应为 4D [B, 1, Freq, T_fixed]
                assert batch_tensor.dim() == 4, (
                    f"[collate_fn.truncate_pad] 特征 '{feat_name}' batch 输出维度错误: "
                    f"期望 4D [B, 1, Freq, T], 实际 {batch_tensor.dim()}D {tuple(batch_tensor.shape)}"
                )

                # 维度检查：填充/截断后长度与 max_frames 一致
                assert batch_tensor.shape[-1] == max_frames, (
                    f"[collate_fn.truncate_pad] 特征 '{feat_name}' 时间维度错误: "
                    f"期望 {max_frames}, 实际 {batch_tensor.shape[-1]}"
                )

                batch_output[feat_name] = batch_tensor

            final_batch = {'x': batch_output}

        elif strategy == "dynamic_mask":
            max_len = lengths.max().item()
            for feat_name in feature_names:
                padded_feats = []
                for idx, feat_dict in enumerate(features_list):
                    feat = feat_dict[feat_name]

                    # 维度检查：单样本输入应为 1D [T] 或 2D [D, T]
                    assert feat.dim() in (1, 2), (
                        f"[collate_fn.dynamic_mask] 样本 {idx} 特征 '{feat_name}' 维度错误: "
                        f"期望 1D [T] 或 2D [D, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
                    )

                    padded_feats.append(F.pad(feat, (0, max_len - feat.shape[-1])))
                stacked_feat = torch.stack(padded_feats)
                # 为 Transformer 转换为 [Batch, Time, Feature_dim]
                if stacked_feat.dim() == 3:
                    stacked_feat = stacked_feat.transpose(1, 2)
                batch_output[feat_name] = stacked_feat

            # 生成全局 Attention Mask: True 表示真实数据，False 表示人工补零区域
            mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

            # 维度检查：mask 与数据时间维度匹配
            assert mask.shape == (len(batch), max_len), (
                f"[collate_fn.dynamic_mask] mask 维度错误: "
                f"期望 ({len(batch)}, {max_len}), 实际 {tuple(mask.shape)}"
            )

            # 维度检查：mask 格式为布尔型
            assert mask.dtype == torch.bool, (
                f"[collate_fn.dynamic_mask] mask 类型错误: 期望 torch.bool, 实际 {mask.dtype}"
            )

            final_batch = {'x': batch_output, 'mask': mask}

        elif strategy == "sliding_window":
            # 用于评估/测试阶段长音频按一定重叠率切分投票
            window_size = proc_cfg.get('window_size', 300)
            stride = proc_cfg.get('stride', 150)

            window_counts = []     # 记录每个原始样本展开了几个窗口，用于推理后复原并求均值
            expanded_labels = []   # 与展开后窗口一一对应的标签集合
            unexpanded_labels = [] # 原始各个样本的标签集合 (原batch尺寸)

            for feat_name in feature_names:
                all_windows = []
                for i, feat_dict in enumerate(features_list):
                    feat = feat_dict[feat_name]

                    # 维度检查：单样本输入应为 1D [T] 或 2D [Freq, T]
                    assert feat.dim() in (1, 2), (
                        f"[collate_fn.sliding_window] 样本 {i} 特征 '{feat_name}' 维度错误: "
                        f"期望 1D [T] 或 2D [Freq, T], 实际 {feat.dim()}D {tuple(feat.shape)}"
                    )

                    current_frames = feat.shape[-1]

                    sample_windows = []
                    # 时长不足整个窗口的直接补零到设定的尺寸
                    if current_frames <= window_size:
                        sample_windows.append(F.pad(feat, (0, window_size - current_frames)))
                    else:
                        # 超过窗口大小的进行沿时间轴切片
                        for start in range(0, current_frames - window_size + 1, stride):
                            sample_windows.append(feat[..., start:start + window_size])

                        # 若结尾有遗漏的数据片段，向反方向补充截取最后一段
                        if (current_frames - window_size) % stride != 0:
                            sample_windows.append(feat[..., -window_size:])

                    all_windows.extend(sample_windows)

                    # 使用遇到第一个特征时结算标签以避免重复计算
                    if feat_name == list(feature_names)[0]:
                        num_win = len(sample_windows)
                        window_counts.append(num_win)
                        expanded_labels.extend([labels[i].item()] * num_win)
                        unexpanded_labels.append(labels[i].item())

                # 为展开后的片段添加 Channel 维度，与 truncate_pad 格式保持一致
                batch_tensor = torch.stack(all_windows).unsqueeze(1)

                # 维度检查：窗口切分后输出应为 4D [B_expanded, 1, Freq, window_size]
                assert batch_tensor.dim() == 4, (
                    f"[collate_fn.sliding_window] 特征 '{feat_name}' batch 输出维度错误: "
                    f"期望 4D [B_expanded, 1, Freq, window_size], 实际 {batch_tensor.dim()}D {tuple(batch_tensor.shape)}"
                )

                # 维度检查：窗口大小与配置一致
                assert batch_tensor.shape[-1] == window_size, (
                    f"[collate_fn.sliding_window] 特征 '{feat_name}' 窗口大小错误: "
                    f"期望 {window_size}, 实际 {batch_tensor.shape[-1]}"
                )

                batch_output[feat_name] = batch_tensor

            labels = torch.tensor(expanded_labels, dtype=torch.long)
            window_counts_tensor = torch.tensor(window_counts, dtype=torch.long)
            original_labels_tensor = torch.tensor(unexpanded_labels, dtype=torch.long)

            # 维度检查：window_counts 与原始 batch 大小匹配
            assert len(window_counts) == len(batch), (
                f"[collate_fn.sliding_window] window_counts 长度错误: "
                f"期望 {len(batch)} (原始 batch 大小), 实际 {len(window_counts)}"
            )

            # 维度检查：original_labels 与原始 batch 大小匹配
            assert len(unexpanded_labels) == len(batch), (
                f"[collate_fn.sliding_window] original_labels 长度错误: "
                f"期望 {len(batch)} (原始 batch 大小), 实际 {len(unexpanded_labels)}"
            )

            # 维度检查：expanded_labels 与展开后窗口总数匹配
            total_windows = sum(window_counts)
            assert len(expanded_labels) == total_windows, (
                f"[collate_fn.sliding_window] expanded_labels 长度错误: "
                f"期望 {total_windows} (窗口总数), 实际 {len(expanded_labels)}"
            )

            final_batch = {
                'x': batch_output,
                'window_counts': window_counts_tensor,
                'original_labels': original_labels_tensor
            }

        else:
            raise ValueError(f"未知的对齐策略: {strategy}")

        # -------------------------------------------------------------
        # 阶段 2: 批处理级数据增强 (Batch-Level Augmentation)
        # 针对【课程 7】的高阶平滑与泛化实践
        # -------------------------------------------------------------
        if enable_mixup:
            # TODO: 教学生利用 Beta 分布计算混合系数 lambda，
            # 获取打乱的批次索引 (torch.randperm)，然后执行输入特征与 One-hot 标签的线性插值。
            raise NotImplementedError("请在 collate_fn 中实现 Mixup 批次特征与软标签融合！")

        return final_batch, labels

    return collate_fn
