# 贡献指南

本仓库只维护可复用的 SER Python 能力，不接受桌面端、Web UI、本地 HTTP
服务或产品工作区代码。提交前请阅读
[`docs/REPOSITORY_IMPLEMENTATION_PLAN.md`](docs/REPOSITORY_IMPLEMENTATION_PLAN.md)。

## 开发环境

```bash
python -m venv .venv
python -m pip install -e ".[test,dev]"
python -m pytest -q
python -m ruff check ser_lib tests data benchmarks scripts examples
python -m mypy --follow-imports=skip ser_lib
python -m build
```

新增公开配置必须禁止未知字段；新增 Representation 必须声明 `TensorSpec`；
新增模型必须实现 `ModelSpec`、`model_config` 和标准 `ModelOutput`。数据格式差异
应通过 importer 或 Representation 扩展，禁止重新加入按 Waveform/MFCC/Mel
拆分的 Dataset。

测试至少覆盖正常路径、无效配置、变长 batch 和序列化往返。新增 importer 还必须
覆盖缺失文件、非法元数据、标签映射和说话人泄漏检查，并使用合成 fixture 测试。
不得提交数据集、模型权重、`runs/`、`artifacts/`、个人路径、密钥或未经许可的媒体。
