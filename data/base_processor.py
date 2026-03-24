import json
import math
import yaml
import torchaudio
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class DatasetProcessor:
    """
    语音情感识别 (SER) 纯净数据预处理基类。
    
    采用模板方法设计模式。子类仅需实现解析具体音频文件和提取元信息的最小逻辑，
    由基类全权负责进度条显示、时长统计、类别映射、数据打乱分割、JSONL 落盘、
    数据分析报表生成以及标准化 YAML 配置文件的一键导出。
    """
    
    def __init__(self, raw_data_dir: str, output_meta_dir: str, dataset_name: str, emotion_mapping: Dict[str, int], emotion_mapping_zh: Dict[str, str] = None):
        """
        初始化处理器。
        
        Args:
            raw_data_dir: 原始变长 WAV 等音频文件所在根目录
            output_meta_dir: 处理后 JSONL 和 YAML 的输出目录
            dataset_name: 数据集名称 (如 "CASIA", "EMOTION_TALK")
            emotion_mapping: 该数据集情感文本到整型 ID 的映射字典字典 (如 {"happy": 1})
            emotion_mapping_zh: 该数据集情感文本到中文描述的映射字典字典 (可选项，如 {"happy": "高兴"})
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_meta_dir = Path(output_meta_dir)
        self.dataset_name = dataset_name
        self.emotion_mapping = emotion_mapping
        self.emotion_mapping_zh = emotion_mapping_zh if emotion_mapping_zh else {}
        
        # 内部统计桩
        self.all_data: List[Dict[str, Any]] = []
        self.total_duration_sec = 0.0
        self.emotion_counts = {k: 0 for k in self.emotion_mapping.keys()}
        self.speaker_counts: Dict[str, int] = {}
        self.duration_seconds: List[float] = []
        self.duration_histogram: List[Dict[str, Any]] = []
        self.duration_percentiles: Dict[str, float] = {}
        self.recommended_audio_processing: Dict[str, Any] = {}

    def _build_base_config(self) -> Dict[str, Any]:
        """构建所有模板共享的基础配置。"""
        class_mapping = {
            label_id: {
                "en": emotion_name.capitalize(),
                "zh": self.emotion_mapping_zh.get(emotion_name, emotion_name),
            }
            for emotion_name, label_id in sorted(self.emotion_mapping.items(), key=lambda item: item[1])
        }

        return {
            "dataset_name": self.dataset_name,
            "num_classes": len(self.emotion_mapping),
            "class_mapping": class_mapping,
            "paths": {
                "metadata_dir": str(self.output_meta_dir.as_posix()),
                "data_root_dir": str(self.raw_data_dir.as_posix()),
            },
            "data_lists": {
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
            },
            "audio": {
                "target_sr": 16000,
            },
            "audio_processing": {
                "strategy": "truncate_pad",
                "max_frames": 300,
                "window_size": 300,
                "stride": 150,
            },
            "transforms": {
                "waveform_level": {"train": [], "val": [], "test": []},
                "advanced_waveform_level": {"train": [], "val": [], "test": []},
                "spectrogram_level": {"train": [], "val": [], "test": []},
                "batch_level": {"train": [], "val": [], "test": []},
            },
        }

    def _build_waveform_dataset_config(self, strategy: str = "dynamic_mask") -> Dict[str, Any]:
        """构建 WaveformDataset 配置。"""
        config = self._build_base_config()
        config["audio_processing"]["strategy"] = strategy
        config["audio_processing"]["max_frames"] = 64000
        config["audio_processing"]["window_size"] = 64000
        config["audio_processing"]["stride"] = 32000
        return config

    def _build_spectrogram_dataset_config(
        self,
        strategy: str = "truncate_pad",
        spectrogram_type: str = "LogMelSpectrogram",
    ) -> Dict[str, Any]:
        """构建 SpectrogramDataset 配置。"""
        config = self._build_base_config()
        config["audio_processing"]["strategy"] = strategy
        config["spectrogram"] = {
            "type": spectrogram_type,
            "kwargs": {
                "sample_rate": 16000,
                "n_fft": 1024,
                "win_length": 1024,
                "hop_length": 256,
                "n_mels": 80,
                "f_min": 0.0,
                "f_max": 8000.0,
                "power": 2.0,
                "top_db": 80.0,
            },
        }
        return config

    def _build_default_dataset_config(
        self,
        template_type: str,
        strategy: Optional[str] = None,
        spectrogram_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建默认模板配置。子类一般无需重写。"""
        if template_type == "waveform":
            return self._build_waveform_dataset_config(strategy or "dynamic_mask")
        if template_type == "spectrogram":
            return self._build_spectrogram_dataset_config(
                strategy or "truncate_pad",
                spectrogram_type or "LogMelSpectrogram",
            )
        if template_type == "feature":
            return self._build_feature_dataset_config(strategy or "dynamic_mask")
        raise ValueError(f"未知模板类型: {template_type}")

    def _build_custom_dataset_config(
        self,
        template_type: str,
        strategy: Optional[str] = None,
        spectrogram_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建自定义模板配置。

        默认行为是回退到默认模板。子类可以覆写该方法，生成更贴合具体数据集的配置。
        """
        return self._build_default_dataset_config(template_type, strategy, spectrogram_type)

    def _build_feature_dataset_config(self, strategy: str = "dynamic_mask") -> Dict[str, Any]:
        """构建 FeatureDataset 配置。"""
        config = self._build_base_config()
        config["audio_processing"]["strategy"] = strategy
        config["features"] = {
            "selected_features": {
                "f0": {
                    "type": "F0",
                    "kwargs": {"hop_length": 256},
                },
                "rms": {
                    "type": "RMS",
                    "kwargs": {"win_length": 400, "hop_length": 256},
                },
                "zcr": {
                    "type": "ZCR",
                    "kwargs": {"win_length": 400, "hop_length": 256},
                },
                "spectral_centroid": {
                    "type": "SpectralCentroid",
                    "kwargs": {"n_fft": 1024, "hop_length": 256},
                },
            }
        }
        return config

    def _summarize_duration_distribution(self, bin_width: float = 0.5) -> List[Dict[str, Any]]:
        """统计音频时长分布及概率。"""
        if not self.duration_seconds:
            return []

        histogram: Dict[int, int] = {}
        for duration in self.duration_seconds:
            bin_index = int(math.floor(duration / bin_width))
            histogram[bin_index] = histogram.get(bin_index, 0) + 1

        total = len(self.duration_seconds)
        summary = []
        for bin_index in sorted(histogram.keys()):
            start = round(bin_index * bin_width, 3)
            end = round((bin_index + 1) * bin_width, 3)
            count = histogram[bin_index]
            probability = count / total
            summary.append(
                {
                    "range": f"[{start:.1f}, {end:.1f})s",
                    "start": start,
                    "end": end,
                    "count": count,
                    "probability": round(probability, 4),
                }
            )

        return summary

    def _compute_percentile(self, percentile: float) -> float:
        """按时长从长到短累计，计算给定百分位数对应的时长（秒）。

        说明：
        - P50 表示从长到短累计 50% 样本时的时长阈值
        - P90 表示从长到短累计 90% 样本时的时长阈值
        - 与常见“从短到长”定义相比，这里等价于使用 (1 - percentile) 的升序分位点
        """
        if not self.duration_seconds:
            return 0.0

        sorted_durations = sorted(self.duration_seconds)
        if len(sorted_durations) == 1:
            return round(sorted_durations[0], 3)

        ascending_percentile = 1.0 - percentile
        position = (len(sorted_durations) - 1) * ascending_percentile
        lower_idx = int(math.floor(position))
        upper_idx = int(math.ceil(position))
        lower = sorted_durations[lower_idx]
        upper = sorted_durations[upper_idx]
        interpolated = lower + (upper - lower) * (position - lower_idx)
        return round(interpolated, 3)

    def _summarize_duration_percentiles(self) -> Dict[str, float]:
        """统计常用时长分位数。"""
        if not self.duration_seconds:
            return {}

        return {
            "p50": self._compute_percentile(0.50),
            "p75": self._compute_percentile(0.75),
            "p90": self._compute_percentile(0.90),
            "p95": self._compute_percentile(0.95),
            "max": round(max(self.duration_seconds), 3),
        }

    def _estimate_recommended_audio_processing(self) -> Dict[str, Any]:
        """基于主体样本分布给出推荐的固定长度参数。

        说明:
        - compact: 更偏向主峰短样本，适合资源敏感场景
        - balanced: 默认推荐，优先覆盖大多数主体样本
        - conservative: 更保守，覆盖更多长尾样本
        """
        if not self.duration_percentiles or not self.duration_histogram:
            return {}

        target_sr = 16000
        hop_length = 256

        dominant_bucket = max(self.duration_histogram, key=lambda item: (item["count"], -item["start"]))
        compact_seconds = self.duration_percentiles["p90"]
        balanced_seconds = self.duration_percentiles["p75"]
        conservative_seconds = self.duration_percentiles["p50"]

        def seconds_to_frames(seconds: float) -> Dict[str, int]:
            return {
                "waveform_max_frames": int(math.ceil(seconds * target_sr)),
                "spectrogram_max_frames": int(math.ceil(seconds * target_sr / hop_length)),
            }

        compact = seconds_to_frames(compact_seconds)
        balanced = seconds_to_frames(balanced_seconds)
        conservative = seconds_to_frames(conservative_seconds)

        return {
            "dominant_range": dominant_bucket["range"],
            "compact_seconds": round(compact_seconds, 3),
            "balanced_seconds": round(balanced_seconds, 3),
            "conservative_seconds": round(conservative_seconds, 3),
            "compact": compact,
            "balanced": balanced,
            "conservative": conservative,
            "target_sr": target_sr,
            "hop_length": hop_length,
            "default_basis": "descending_p75",
        }

    def _print_duration_distribution(self):
        """在控制台打印时长分布概览。"""
        if not self.duration_histogram:
            print("   - 时长分布：暂无可用统计")
            return

        print("   - 时长分布（按 0.5s 分桶）：")
        for item in self.duration_histogram:
            probability_pct = item["probability"] * 100
            print(
                f"     * {item['range']}: {item['count']} 条 "
                f"({probability_pct:.1f}%)"
            )

    def _print_duration_percentiles(self):
        """在控制台打印时长分位数与推荐配置。"""
        if not self.duration_percentiles:
            print("   - 时长分位数：暂无可用统计")
            return

        print(
            "   - 时长分位数（从长到短累计）: "
            f"P50={self.duration_percentiles['p50']:.3f}s, "
            f"P75={self.duration_percentiles['p75']:.3f}s, "
            f"P90={self.duration_percentiles['p90']:.3f}s, "
            f"P95={self.duration_percentiles['p95']:.3f}s, "
            f"Max={self.duration_percentiles['max']:.3f}s"
        )

        if self.recommended_audio_processing:
            print(
                "   - 主峰时长区间（概率最高）: "
                f"{self.recommended_audio_processing['dominant_range']}"
            )
            print(
                "   - 推荐固定长度参数: "
                f"compact={self.recommended_audio_processing['compact_seconds']:.3f}s "
                f"(waveform={self.recommended_audio_processing['compact']['waveform_max_frames']}, "
                f"spectrogram={self.recommended_audio_processing['compact']['spectrogram_max_frames']}), "
                f"balanced={self.recommended_audio_processing['balanced_seconds']:.3f}s "
                f"(waveform={self.recommended_audio_processing['balanced']['waveform_max_frames']}, "
                f"spectrogram={self.recommended_audio_processing['balanced']['spectrogram_max_frames']}), "
                f"conservative={self.recommended_audio_processing['conservative_seconds']:.3f}s "
                f"(waveform={self.recommended_audio_processing['conservative']['waveform_max_frames']}, "
                f"spectrogram={self.recommended_audio_processing['conservative']['spectrogram_max_frames']})"
            )

    def _prompt_choice(self, title: str, options: Dict[str, str], default: str) -> str:
        """通过命令行交互选择配置项。

        支持三种输入方式:
        1. 直接输入完整 key
        2. 输入从 1 开始的序号
        3. 输入能唯一匹配的 key 前缀缩写
        """
        print(f"\n{title}")
        option_keys = list(options.keys())
        for index, (key, desc) in enumerate(options.items(), start=1):
            default_marker = " (默认)" if key == default else ""
            print(f"  {index}. {key} - {desc}{default_marker}")

        raw = input(f"请输入选项 [{default}]: ").strip().lower()
        if not raw:
            return default

        if raw in options:
            return raw

        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(option_keys):
                return option_keys[index]

        matched_keys = [key for key in option_keys if key.lower().startswith(raw)]
        if len(matched_keys) == 1:
            return matched_keys[0]
        if len(matched_keys) > 1:
            print(f"⚠️ 输入 '{raw}' 同时匹配 {matched_keys}，将使用默认值 {default}。")
            return default

        print(f"⚠️ 无效输入 '{raw}'，将使用默认值 {default}。")
        return default

    def _build_dataset_config(
        self,
        config_mode: str,
        template_type: str,
        strategy: Optional[str] = None,
        spectrogram_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """根据模式与模板类型构建与当前 schema 兼容的配置。"""
        if config_mode == "default":
            return self._build_default_dataset_config(template_type, strategy, spectrogram_type)
        if config_mode == "preset":
            return self._build_custom_dataset_config(template_type, strategy, spectrogram_type)
        if config_mode == "manual":
            return self._build_default_dataset_config(template_type, strategy, spectrogram_type)
        raise ValueError(f"未知配置模式: {config_mode}")

    def _prompt_int(self, prompt: str, default: int) -> int:
        """交互输入整数，支持回车使用默认值。"""
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"⚠️ 无效整数 '{raw}'，将使用默认值 {default}。")
            return default

    def _prompt_float(self, prompt: str, default: float) -> float:
        """交互输入浮点数，支持回车使用默认值。"""
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"⚠️ 无效浮点数 '{raw}'，将使用默认值 {default}。")
            return default

    def _apply_manual_overrides(
        self,
        config: Dict[str, Any],
        template_type: str,
        strategy: str,
    ) -> Dict[str, Any]:
        """在 manual 模式下交互覆盖关键配置参数。"""
        print("\n进入 manual 模式：你可以手动覆盖关键配置参数。直接回车表示使用当前值。")

        audio_cfg = config.setdefault("audio", {})
        audio_cfg["target_sr"] = self._prompt_int("请输入 target_sr", int(audio_cfg.get("target_sr", 16000)))

        proc_cfg = config.setdefault("audio_processing", {})
        proc_cfg["strategy"] = strategy

        recommendation_choice = None
        if self.recommended_audio_processing and strategy in {"truncate_pad", "sliding_window"}:
            recommendation_choice = self._prompt_choice(
                "是否使用时长统计推荐值预填长度参数？",
                {
                    "manual": "不使用推荐值，完全手动输入",
                    "compact": "使用 compact 推荐值（更短、更省资源）",
                    "balanced": "使用 balanced 推荐值（默认推荐）",
                    "conservative": "使用 conservative 推荐值（更长、更保守）",
                },
                default="balanced",
            )

        def resolve_recommended_frame(key: str, current_default: int) -> int:
            if recommendation_choice in {"compact", "balanced", "conservative"}:
                return int(self.recommended_audio_processing[recommendation_choice][key])
            return current_default

        if strategy == "truncate_pad":
            proc_cfg["max_frames"] = self._prompt_int(
                "请输入 max_frames",
                resolve_recommended_frame("spectrogram_max_frames" if template_type == "spectrogram" else "waveform_max_frames", int(proc_cfg.get("max_frames", 300))),
            )
        elif strategy == "sliding_window":
            proc_cfg["window_size"] = self._prompt_int(
                "请输入 window_size",
                resolve_recommended_frame("spectrogram_max_frames" if template_type == "spectrogram" else "waveform_max_frames", int(proc_cfg.get("window_size", 300))),
            )
            proc_cfg["stride"] = self._prompt_int(
                "请输入 stride",
                int(proc_cfg.get("stride", 150)),
            )
        else:
            print("dynamic_mask 模式不强制要求固定长度参数，保留当前默认值。")

        if template_type == "spectrogram":
            spec_cfg = config.setdefault("spectrogram", {})
            kwargs = spec_cfg.setdefault("kwargs", {})
            kwargs["sample_rate"] = audio_cfg["target_sr"]
            kwargs["n_fft"] = self._prompt_int("请输入 spectrogram.n_fft", int(kwargs.get("n_fft", 1024)))
            kwargs["win_length"] = self._prompt_int("请输入 spectrogram.win_length", int(kwargs.get("win_length", kwargs["n_fft"])))
            kwargs["hop_length"] = self._prompt_int("请输入 spectrogram.hop_length", int(kwargs.get("hop_length", 256)))
            if spec_cfg.get("type") in {"MelSpectrogram", "LogMelSpectrogram", "MFCC"}:
                kwargs["n_mels"] = self._prompt_int("请输入 spectrogram.n_mels", int(kwargs.get("n_mels", 80)))
            if spec_cfg.get("type") == "MFCC":
                kwargs["n_mfcc"] = self._prompt_int("请输入 spectrogram.n_mfcc", int(kwargs.get("n_mfcc", 40)))
            if spec_cfg.get("type") == "LogMelSpectrogram":
                kwargs["top_db"] = self._prompt_float("请输入 spectrogram.top_db", float(kwargs.get("top_db", 80.0)))

        if template_type == "feature":
            selected = config.get("features", {}).get("selected_features", {})
            for feat_name, feat_cfg in selected.items():
                kwargs = feat_cfg.setdefault("kwargs", {})
                if "hop_length" in kwargs:
                    kwargs["hop_length"] = self._prompt_int(
                        f"请输入特征 {feat_name}.hop_length",
                        int(kwargs.get("hop_length", 256)),
                    )
                if "win_length" in kwargs:
                    kwargs["win_length"] = self._prompt_int(
                        f"请输入特征 {feat_name}.win_length",
                        int(kwargs.get("win_length", 400)),
                    )
                if "n_fft" in kwargs:
                    kwargs["n_fft"] = self._prompt_int(
                        f"请输入特征 {feat_name}.n_fft",
                        int(kwargs.get("n_fft", 1024)),
                    )

        return config

    def _build_dataset_config_interactive(self) -> Dict[str, Any]:
        """通过命令行交互式生成配置。"""
        config_mode = self._prompt_choice(
            "请选择配置生成模式:",
            {
                "default": "默认模板 - 使用框架推荐初始配置",
                "preset": "预设模板 - 使用子类定义的数据集专属参数",
                "manual": "手动模板 - 逐项输入关键参数",
            },
            default="default",
        )

        template_type = self._prompt_choice(
            "请选择要生成的数据集模板类型:",
            {
                "waveform": "WaveformDataset - 原始波形输入",
                "spectrogram": "SpectrogramDataset - 单谱图输入",
                "feature": "FeatureDataset - 多特征输入",
            },
            default="spectrogram",
        )

        strategy_options = {
            "waveform": {
                "dynamic_mask": "动态补齐，适合波形时序模型",
                "truncate_pad": "固定长度补齐/截断",
                "sliding_window": "滑窗展开，适合长音频评估",
            },
            "spectrogram": {
                "truncate_pad": "固定长度，适合 2D CNN",
                "dynamic_mask": "动态补齐，适合时序模型",
                "sliding_window": "滑窗展开，适合长音频评估",
            },
            "feature": {
                "dynamic_mask": "动态补齐，适合多特征时序建模",
                "truncate_pad": "固定长度输入",
            },
        }
        strategy_defaults = {
            "waveform": "dynamic_mask",
            "spectrogram": "truncate_pad",
            "feature": "dynamic_mask",
        }
        strategy = self._prompt_choice(
            "请选择 batch 对齐策略:",
            strategy_options[template_type],
            default=strategy_defaults[template_type],
        )

        spectrogram_type = None
        if template_type == "spectrogram":
            spectrogram_type = self._prompt_choice(
                "请选择谱图类型:",
                {
                    "Spectrogram": "普通谱图",
                    "MelSpectrogram": "Mel 频谱图",
                    "LogMelSpectrogram": "Log-Mel 频谱图",
                    "MFCC": "MFCC 特征",
                },
                default="LogMelSpectrogram",
            )

        config = self._build_dataset_config(config_mode, template_type, strategy, spectrogram_type)
        if config_mode == "manual":
            config = self._apply_manual_overrides(config, template_type, strategy)
        config["_template_type"] = template_type
        config["_config_mode"] = config_mode
        return config
        
    def _extract_samples(self) -> List[Dict[str, Any]]:
        """
        【必须由子类实现】
        遍历 `self.raw_data_dir`，解析文件名或附带的标注文件。
        
        Expected Return:
            返回一个完整元数据字典列表。每个字典必须包含且结构如下：
            {
                "audio_path": "绝对路径字符串, 需替换 \\ 为 /",
                "label": int,           # 查 emotion_mapping 获得
                "emotion_text": str,    # 原始情感英文小写
                "speaker_id": str,      # 说话人唯一标识
            }
            可扩展参数: "start_time_ms", "end_time_ms", "text" 等。
            *注意*: 不在此计算 duration 时长，由基类的统一进度条管辖计算！
        """
        raise NotImplementedError("子类必须实现具体的音频树解析逻辑：_extract_samples()")

    def _split_strategy(self, data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        【必须由子类实现】
        定义训练、验证、测试集的硬拆分策略。
        
        CASIA 等孤立词可用 Speaker-wise 切分，而有剧本的对话长语音可能需要 Session-wise 切分。
        
        Args:
            data_list: 已包含 duration 并经过初始乱序过滤的元数据全量列表。
        
        Return:
            字典形式的分组列表，必定包含 'train', 'val', 'test' 键。
        """
        raise NotImplementedError("子类必须实现数据切分逻辑：_split_strategy()")

    def process(self):
        """执行全套预处理标准生命周期流水线。"""
        print(f"\n[{self.dataset_name}] 开始构建抽象解析管线...")
        self.output_meta_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 调用子类业务剥离元数据
        raw_samples = self._extract_samples()
        if not raw_samples:
            print("未找到任何符合条件的样本，已中断后续链路。")
            return
            
        print(f"✅ 子类抽取结束，发现有效待测元数据 {len(raw_samples)} 条，开始全局声学校验与耗时测算：")
        
        # 2. 挂载全局 TQDM 进度条测算音频时长 (剔除破损文件)
        valid_samples = []
        for item in tqdm(raw_samples, desc="声学指标校验", unit="file"):
            abs_path = item["audio_path"]
            try:
                info = torchaudio.info(abs_path)
                duration = info.num_frames / info.sample_rate
                
                item["duration"] = round(duration, 3)
                valid_samples.append(item)
                self.duration_seconds.append(duration)
                
                # 更新全局内部统计钩子
                self.total_duration_sec += duration
                self.emotion_counts[item["emotion_text"]] += 1
                spk = item["speaker_id"]
                self.speaker_counts[spk] = self.speaker_counts.get(spk, 0) + 1
                
            except Exception as e:
                # 记录但不抛出异常
                print(f"⚠️ 跳过无效音频: {abs_path} | 原因: {e}")
                
        self.all_data = valid_samples
        self.duration_histogram = self._summarize_duration_distribution()
        self.duration_percentiles = self._summarize_duration_percentiles()
        self.recommended_audio_processing = self._estimate_recommended_audio_processing()
        print(f"📊 {self.dataset_name} 数据画像统揽：")
        print(f"   - 有效音频：{len(self.all_data)} 句")
        print(f"   - 总计时长：{self.total_duration_sec / 3600:.2f} 小时")
        print(f"   - 发音人数：{len(self.speaker_counts)} 人")
        self._print_duration_distribution()
        self._print_duration_percentiles()
        
        # 3. 数据隔离拆分
        splits = self._split_strategy(self.all_data)
        
        # 4. JSONL 持久化写入
        self._write_jsonls(splits)
        
        # 5. 分析报告写入
        self._write_markdown_report(splits)
        
        # 6. 生成可直接被系统加载的基础环境 YAML
        self._generate_project_yaml()
        
        print(f"\n🎉 {self.dataset_name} 全流水线组装完成，就绪！")

    def _write_jsonls(self, splits: Dict[str, List[Dict[str, Any]]]):
        """序列化写入"""
        print("\n正在序列化切割数据 -> JSONL...")
        for split_name, data in splits.items():
            jsonl_path = self.output_meta_dir / f"{split_name}.jsonl"
            # 统计分布
            split_dist = {k: 0 for k in self.emotion_mapping.keys()}
            for item in data:
                split_dist[item["emotion_text"]] += 1
                
            dist_str = ", ".join([f"{k}:{v}" for k,v in split_dist.items() if v > 0])
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"   └─ [{split_name.upper():5}] : {len(data):4} 条 | 分布: [{dist_str}] -> {jsonl_path.name}")

    def _write_markdown_report(self, splits: Dict[str, List[Dict[str, Any]]]):
        """生成标准化 Markdown 降级视图报告"""
        report_path = self.output_meta_dir / "data_report.md"
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write(f"# {self.dataset_name} 数据集工程化诊断报表\n\n")
            rf.write("## 1. 总体概况\n")
            rf.write(f"- **有效音频数**: {len(self.all_data)} 条\n")
            rf.write(f"- **总发声时长**: {self.total_duration_sec / 3600:.2f} 小时 ({self.total_duration_sec:.2f} 秒)\n")
            rf.write(f"- **覆盖说话人数**: {len(self.speaker_counts)} 人\n\n")
            
            rf.write("## 1.1 时长分布统计\n")
            if self.duration_histogram:
                for item in self.duration_histogram:
                    rf.write(
                        f"- **{item['range']}**: {item['count']} 条 "
                        f"(概率 {item['probability'] * 100:.1f}%)\n"
                    )
            else:
                rf.write("- 无可用时长统计\n")
            rf.write("\n")

            rf.write("## 1.2 时长分位数与推荐固定长度\n")
            if self.duration_percentiles:
                rf.write("- **分位数口径**: 从长到短累计统计\n")
                rf.write(
                    f"- **P50**: {self.duration_percentiles['p50']:.3f}s\n"
                    f"- **P75**: {self.duration_percentiles['p75']:.3f}s\n"
                    f"- **P90**: {self.duration_percentiles['p90']:.3f}s\n"
                    f"- **P95**: {self.duration_percentiles['p95']:.3f}s\n"
                    f"- **Max**: {self.duration_percentiles['max']:.3f}s\n"
                )
                if self.recommended_audio_processing:
                    rf.write(
                        f"- **主峰时长区间**: {self.recommended_audio_processing['dominant_range']}\n"
                        f"- **Compact 推荐**: {self.recommended_audio_processing['compact_seconds']:.3f}s | "
                        f"Waveform max_frames={self.recommended_audio_processing['compact']['waveform_max_frames']} | "
                        f"Spectrogram max_frames={self.recommended_audio_processing['compact']['spectrogram_max_frames']}\n"
                        f"- **Balanced 推荐（默认优先）**: {self.recommended_audio_processing['balanced_seconds']:.3f}s | "
                        f"Waveform max_frames={self.recommended_audio_processing['balanced']['waveform_max_frames']} | "
                        f"Spectrogram max_frames={self.recommended_audio_processing['balanced']['spectrogram_max_frames']}\n"
                        f"- **Conservative 推荐**: {self.recommended_audio_processing['conservative_seconds']:.3f}s | "
                        f"Waveform max_frames={self.recommended_audio_processing['conservative']['waveform_max_frames']} | "
                        f"Spectrogram max_frames={self.recommended_audio_processing['conservative']['spectrogram_max_frames']}\n"
                        f"- **换算参数**: target_sr={self.recommended_audio_processing['target_sr']}, "
                        f"hop_length={self.recommended_audio_processing['hop_length']}\n"
                    )
            else:
                rf.write("- 无可用时长分位数统计\n")
            rf.write("\n")
            
            rf.write("## 2. 靶向标签分布\n")
            for emo, count in self.emotion_counts.items():
                if count > 0:
                    rf.write(f"- **{emo.capitalize()}** (ID {self.emotion_mapping[emo]}): {count} 频次 (占比 {count/len(self.all_data)*100:.1f}%)\n")
            rf.write("\n")
            
            rf.write("## 3. 切分边界及类别分布\n")
            for split_name in ['train', 'val', 'test']:
                if split_name in splits:
                    data = splits[split_name]
                    split_dist = {k: 0 for k in self.emotion_mapping.keys()}
                    for item in data:
                        split_dist[item["emotion_text"]] += 1
                        
                    rf.write(f"- **{split_name.capitalize()} 域**: 拦截包含 {len(data)} 条语句。\n")
                    rf.write(f"  - 分布情况: ")
                    dist_str = ", ".join([f"{k}: {v} ({v/len(data)*100:.1f}%)" for k, v in split_dist.items() if v > 0])
                    rf.write(f"{dist_str}\n\n")

    def _generate_project_yaml(self, interactive: bool = True):
        """生成与当前 dataset schema 兼容的运行时 YAML 配置文件。"""
        yaml_config_dir = self.output_meta_dir
        yaml_config_dir.mkdir(parents=True, exist_ok=True)
        if interactive:
            config_dict = self._build_dataset_config_interactive()
        else:
            config_dict = self._build_default_dataset_config("spectrogram")

        template_type = config_dict.pop("_template_type", "spectrogram")
        config_dict.pop("_config_mode", None)
        yaml_filename = f"{self.dataset_name.lower()}_{template_type}.yaml"
        yaml_path = yaml_config_dir / yaml_filename
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
        print(f"[配置中心] 标准环境描述映射文件已输出至 -> {yaml_path}")
