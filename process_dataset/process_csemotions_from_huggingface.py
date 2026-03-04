import os
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# 1. 设置保存路径
output_dir = "./CSEMOTIONS-data/wav_data"  # 保存音频的文件夹
csv_path = "./CSEMOTIONS-data/csemotions_metadata.csv"  # 保存信息的 CSV 文件路径

# 创建存放音频的文件夹（如果不存在的话）
os.makedirs(output_dir, exist_ok=True)

# 2. 加载数据集 (由于之前已经缓存过，这里会瞬间加载完毕)
print("正在加载数据集...")
dataset = load_dataset("AIDC-AI/CSEMOTIONS", cache_dir="huggingface_data")
train_data = dataset['train']

# 用于临时存放 CSV 数据的列表
metadata_records = []

print(f"开始导出音频，共 {len(train_data)} 条...")

# 3. 遍历数据集并保存
# tqdm 用于显示进度条，让你直观看到处理到了第几个
for i, sample in enumerate(tqdm(train_data)):
    # 提取信息
    audio_data = sample['audio']
    text = sample['text']
    emotion = sample['emotion']
    speaker = sample['speaker']
    
    # 提取音频波形和采样率
    audio_array = audio_data['array']
    sr = audio_data['sampling_rate']
    
    # 生成一个规范的文件名，例如: 00001_speaker_emotion.wav
    # 为了防止特殊字符导致路径报错，简单的拼装是最好的
    file_name = f"{i:05d}_{speaker}_{emotion}.wav"
    save_path = os.path.join(output_dir, file_name)
    
    # 保存音频为 wav 文件
    # subtype='PCM_24' 严格按照原数据集主页说的 24-bit 精度进行保存
    sf.write(save_path, audio_array, sr, subtype='PCM_24')
    
    # 计算音频时长 (方便后续统计)
    duration = len(audio_array) / sr
    
    # 将当前这条数据的信息追加到列表中
    metadata_records.append({
        "file_name": file_name,
        "text": text,
        "emotion": emotion,
        "speaker": speaker,
        "duration_sec": round(duration, 2)  # 保留两位小数
    })

# 4. 将列表转换为 Pandas DataFrame，并导出为 CSV
print("正在生成 CSV 描述文件...")
df = pd.DataFrame(metadata_records)

# encoding='utf-8-sig' 可以防止包含中文的 text 在 Windows Excel 中打开时乱码
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"🎉 全部导出完成！")
print(f"音频保存在: {os.path.abspath(output_dir)}")
print(f"CSV文件保存在: {os.path.abspath(csv_path)}")