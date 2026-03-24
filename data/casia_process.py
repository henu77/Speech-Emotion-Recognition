import sys
import random
from pathlib import Path

# 因为不使用 __init__.py 且要求绝对导入，而当前文件又作为脚本被直接运行，
# 我们将项目根目录加入 sys.path 以支持从项目根部的绝对导入。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any
from data.base_processor import DatasetProcessor

# CASIA 包含 6 种情感 (有些版本会包含 neutral 或更多，这里采用最经典的 6 分类版本：愤怒、悲伤、高兴、惊吓、恐惧、中性)
EMOTION_MAPPING = {
    'neutral': 0,
    'happy': 1,
    'angry': 2,
    'sad': 3,
    'surprise': 4,
    'fear': 5
}

EMOTION_MAPPING_ZH = {
    'neutral': '平静',
    'happy': '高兴',
    'angry': '愤怒',
    'sad': '悲伤',
    'surprise': '惊吓',
    'fear': '恐惧'
}

class CasiaProcessor(DatasetProcessor):
    def __init__(self, raw_data_dir: str, output_meta_dir: str):
        super().__init__(
            raw_data_dir=raw_data_dir,
            output_meta_dir=output_meta_dir,
            dataset_name="CASIA",
            emotion_mapping=EMOTION_MAPPING,
            emotion_mapping_zh=EMOTION_MAPPING_ZH
        )

    def _extract_samples(self) -> List[Dict[str, Any]]:
        raw_samples = []
        
        # 遍历说话人文件夹
        for speaker_dir in self.raw_data_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            # 遍历情感文件夹
            for emotion_dir in speaker_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                    
                emotion = emotion_dir.name.lower()
                if emotion not in self.emotion_mapping:
                    print(f"警告: 发现未知情感文件夹 {emotion}，已跳过。")
                    continue
                    
                label = self.emotion_mapping[emotion]
                
                # 遍历音频文件
                for audio_file in emotion_dir.glob("*.wav"):
                    # 强转正斜杠，适配跨平台 JSONL
                    abs_path = str(audio_file.resolve()).replace('\\', '/')
                    
                    item = {
                        "audio_path": abs_path,  
                        "label": label,
                        "emotion_text": emotion,
                        "speaker_id": speaker_id
                        # 时长 duration 交由基类计算
                    }
                    raw_samples.append(item)
                    
        return raw_samples

    def _split_strategy(self, data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        # 提取所有说话人并排序保证一致性
        speakers = list(set([item['speaker_id'] for item in data_list]))
        speakers.sort() 
        
        if len(speakers) < 2:
            raise ValueError("CASIA 说话人数量不足 2，无法进行隔离划分！")
            
        # 前 N-1 个作为训练集，最后一个说话人平分给验证和测试集 (Course 5: 隔离评估要求)
        train_speakers = speakers[:-1]
        test_speaker = speakers[-1]
        
        train_data = [item for item in data_list if item['speaker_id'] in train_speakers]
        eval_data = [item for item in data_list if item['speaker_id'] == test_speaker]
        
        # 验证集和测试集平分
        random.seed(42)
        random.shuffle(eval_data)
        half = len(eval_data) // 2
        
        return {
            'train': train_data,
            'val': eval_data[:half],
            'test': eval_data[half:]
        }

    def _build_custom_dataset_config(
        self,
        template_type: str,
        strategy: str | None = None,
        spectrogram_type: str | None = None,
    ) -> Dict[str, Any]:
        """CASIA 的自定义模板：给出更贴合中文短句情感识别的默认参数。"""
        config = super()._build_custom_dataset_config(template_type, strategy, spectrogram_type)

        if template_type == "waveform":
            config["audio_processing"]["strategy"] = strategy or "dynamic_mask"
            config["audio_processing"]["max_frames"] = 48000
            config["audio_processing"]["window_size"] = 48000
            config["audio_processing"]["stride"] = 24000
        elif template_type == "spectrogram":
            config["audio_processing"]["strategy"] = strategy or "truncate_pad"
            config["audio_processing"]["max_frames"] = 200
            config["audio_processing"]["window_size"] = 200
            config["audio_processing"]["stride"] = 100
            config["spectrogram"]["type"] = spectrogram_type or "LogMelSpectrogram"
            config["spectrogram"]["kwargs"].update(
                {
                    "sample_rate": 16000,
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 256,
                    "n_mels": 80,
                    "top_db": 80.0,
                }
            )
        elif template_type == "feature":
            config["audio_processing"]["strategy"] = strategy or "dynamic_mask"
            config["audio_processing"]["max_frames"] = 200
            config["features"]["selected_features"] = {
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

        return config

if __name__ == "__main__":
    # CASIA 物理存放位置
    CASIA_RAW_DIR = "data/CASIA" 
    
    # 抽取出的 JSONL 元数据及 Markdown 报告存放点
    OUTPUT_META_DIR = "ser_lib/dataset/configs/casia"
    
    print("🚀 启动 CASIA 对象化数据洗牌任务...")
    
    if Path(CASIA_RAW_DIR).exists():
        processor = CasiaProcessor(CASIA_RAW_DIR, OUTPUT_META_DIR)
        processor.process()
    else:
        print(f"❌ 严重错误: {CASIA_RAW_DIR} 未找到有效数据卷！")
