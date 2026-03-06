import os
import json
import random
import yaml
from pathlib import Path

# CASIA 包含 6 种情感 (有些版本会包含 neutral 或更多，这里采用最经典的 6 分类版本：愤怒、悲伤、高兴、惊吓、恐惧、中性)
EMOTION_MAPPING = {
    'neutral': 0,
    'happy': 1,
    'angry': 2,
    'sad': 3,
    'surprise': 4,
    'fear': 5
}

def preprocess_casia(raw_data_dir: str, output_dir: str, split_ratio: tuple = (0.8, 0.1, 0.1)):
    """
    处理 CASIA 语音数据集。
    """
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    import torchaudio
    total_duration_sec = 0.0
    emotion_counts = {k: 0 for k in EMOTION_MAPPING.keys()}
    speaker_counts = {}

    all_data = []

    # 遍历说话人文件夹 
    for speaker_dir in raw_path.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        if speaker_id not in speaker_counts:
            speaker_counts[speaker_id] = 0
            
        # 遍历情感文件夹
        for emotion_dir in speaker_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
                
            emotion = emotion_dir.name.lower()
            if emotion not in EMOTION_MAPPING:
                print(f"警告: 发现未知情感文件夹 {emotion}，已跳过。")
                continue
                
            label = EMOTION_MAPPING[emotion]
            
            # 遍历音频文件
            for audio_file in emotion_dir.glob("*.wav"):
                # 获取绝对路径并强制使用 '/' 作为路径分隔符
                abs_path = str(audio_file.resolve()).replace('\\', '/')
                
                # 获取音频时长用于统计
                try:
                    info = torchaudio.info(abs_path)
                    duration = info.num_frames / info.sample_rate
                    total_duration_sec += duration
                except Exception as e:
                    print(f"无法读取音频信息: {abs_path}, 错误: {e}")
                    duration = 0.0
                
                item = {
                    "audio_path": abs_path,  
                    "label": label,
                    "emotion_text": emotion,
                    "speaker_id": speaker_id,
                    "duration": round(duration, 3)
                }
                all_data.append(item)
                
                emotion_counts[emotion] += 1
                speaker_counts[speaker_id] += 1
                
    if not all_data:
        print("未找到任何符合条件的 wav 文件，请检查目录结构！")
        return
        
    print(f"总计找到 {len(all_data)} 条音频样本, 总时长: {total_duration_sec / 3600:.2f} 小时。")
    
    # 统计说话人并进行划分 (前 N-1 个为训练集，最后一个说话人平分给验证和测试集)
    speakers = list(set([item['speaker_id'] for item in all_data]))
    speakers.sort() # 保证顺序一致
    
    if len(speakers) < 2:
        print("说话人数量不足 2，无法进行隔离划分！")
        return
        
    train_speakers = speakers[:-1]
    test_speaker = speakers[-1]
    
    train_data = [item for item in all_data if item['speaker_id'] in train_speakers]
    eval_data = [item for item in all_data if item['speaker_id'] == test_speaker]
    
    # 验证集和测试集平分
    random.seed(42)
    random.shuffle(eval_data)
    half = len(eval_data) // 2
    
    splits = {
        'train': train_data,
        'val': eval_data[:half],
        'test': eval_data[half:]
    }
    
    # 1. 写入 JSONL
    for split_name, data in splits.items():
        jsonl_path = out_path / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                # 必须加 ensure_ascii=False，否则中文路径会被转码为 \uXXXX
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"[{split_name.upper()}] 集划分完成 -> {jsonl_path}")

    # 2. 生成数据统计报表 (data_report.md)
    report_path = out_path / "data_report.md"
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write("# CASIA 数据集统计报表\n\n")
        rf.write("## 1. 总体概况\n")
        rf.write(f"- **总音频数**: {len(all_data)} 条\n")
        rf.write(f"- **总时长**: {total_duration_sec / 3600:.2f} 小时 ({total_duration_sec:.2f} 秒)\n")
        rf.write(f"- **总说话人数**: {len(speakers)} 人\n\n")
        
        rf.write("## 2. 类别分布\n")
        for emo, count in emotion_counts.items():
            rf.write(f"- **{emo.capitalize()}**: {count} 条 ({count/len(all_data)*100:.1f}%)\n")
        rf.write("\n")
        
        rf.write("## 3. 说话人数据分布\n")
        for spk, count in speaker_counts.items():
            role = "Train" if spk in train_speakers else "Val/Test"
            rf.write(f"- **{spk}** [{role}]: {count} 条\n")
        rf.write("\n")
        
        rf.write("## 4. 数据划分情况\n")
        rf.write(f"- **Train**: {len(splits['train'])} 条 (说话人: {', '.join(train_speakers)})\n")
        rf.write(f"- **Val**: {len(splits['val'])} 条 (说话人: {test_speaker})\n")
        rf.write(f"- **Test**: {len(splits['test'])} 条 (说话人: {test_speaker})\n")
        
    print(f"\n[成功] CASIA 详细数据报表已生成 -> {report_path}")

    # 3. 自动生成 YAML 配置
    yaml_config_dir = Path("ser_lib/data/dataset_configs")
    yaml_config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_config_dir / "casia.yaml"
    
    config_dict = {
        "dataset_name": "CASIA",
        "num_classes": len(EMOTION_MAPPING),
        "class_mapping": {v: k.capitalize() for k, v in EMOTION_MAPPING.items()},
        "paths": {
            "metadata_dir": str(out_path.as_posix()),
            "data_root_dir": str(raw_path.as_posix()) 
        },
        "data_lists": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl"
        },
        "audio": {
            "target_sr": 16000,
            "max_duration": 5.0
        },
        "transforms": {
            "train": [
                {"type": "AddNoise", "snr": 15}
            ],
            "val": [{"type": "Normalize"}],
            "test": [{"type": "Normalize"}]
        }
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
    print(f"[成功] CASIA YAML 配置文件已自动生成 -> {yaml_path}")




if __name__ == "__main__":
    CASIA_RAW_DIR = "data/CASIA" # 修改为你的实际路径
    OUTPUT_META_DIR = "ser_lib/datasets/configs/casia/"
    
    print("正在测试 CASIA 处理流程...")
    if Path(CASIA_RAW_DIR).exists():
        preprocess_casia(CASIA_RAW_DIR, OUTPUT_META_DIR)
    else:
        print(f"数据目录 {CASIA_RAW_DIR} 不存在。")
