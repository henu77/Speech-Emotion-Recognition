import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ser_lib.datasets.base_dataset import BaseConfigDataset
    ds = BaseConfigDataset('ser_lib/datasets/configs/casia.yaml')
    print("✅ 成功加载 Dataset！")
    print(f"📦 当前挂载的特征提取器: {list(ds.feature_extractors.keys())}")
    
    # 抽取第0个数据看看工作是否正常
    feats, label, length = ds[0]
    print(f"🎯 第0个样本标签: {label}, 有效序列长度: {length}")
    for fname, tensor in feats.items():
        print(f"   - 特征 [{fname}] shape: {tensor.shape}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
