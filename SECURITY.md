# 安全说明

## 报告问题

请通过代码托管平台的私密安全报告渠道提交漏洞，不要在公开 issue 中披露可利用
细节。维护者确认后会评估受影响版本并协调修复与披露。

## 信任边界

- 分发模型应使用 safetensors artifact；加载时会校验全部组成文件的 SHA-256。
- `.pt` checkpoint 使用 Python pickle，只能加载由自己生成且来源可信的本地文件。
- 旧 artifact 的 pickle 权重必须显式设置 `allow_legacy_pickle=True`。
- Hugging Face 适配器默认仅加载本地文件并强制 `trust_remote_code=False`。
- importer 不执行数据集中的脚本，也不自动下载受限语料。
- 本库不提供执行任意用户 Python 代码的能力。

模型、数据集和依赖各自可能有独立许可与供应链风险，部署者仍须自行审查来源、
哈希、模型卡和使用条款。
