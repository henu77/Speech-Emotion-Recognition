import json
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
        if config_mode == "custom":
            return self._build_custom_dataset_config(template_type, strategy, spectrogram_type)
        raise ValueError(f"未知配置模式: {config_mode}")

    def _build_dataset_config_interactive(self) -> Dict[str, Any]:
        """通过命令行交互式生成配置。"""
        config_mode = self._prompt_choice(
            "请选择配置生成模式:",
            {
                "default": "默认模板 - 使用框架推荐初始配置",
                "custom": "自定义模板 - 允许子类生成数据集专属配置",
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
                
                # 更新全局内部统计钩子
                self.total_duration_sec += duration
                self.emotion_counts[item["emotion_text"]] += 1
                spk = item["speaker_id"]
                self.speaker_counts[spk] = self.speaker_counts.get(spk, 0) + 1
                
            except Exception as e:
                # 记录但不抛出异常
                print(f"⚠️ 跳过无效音频: {abs_path} | 原因: {e}")
                
        self.all_data = valid_samples
        print(f"📊 {self.dataset_name} 数据画像统揽：")
        print(f"   - 有效音频：{len(self.all_data)} 句")
        print(f"   - 总计时长：{self.total_duration_sec / 3600:.2f} 小时")
        print(f"   - 发音人数：{len(self.speaker_counts)} 人")
        
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
        yaml_config_dir = Path("ser_lib/dataset/configs")
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
