# 安装与环境

支持 Python 3.10–3.12，CI 覆盖 Windows 和 Linux，macOS 通过纯 Python 包结构
支持但发布前仍需单独 smoke test。

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

预训练编码器是可选能力：

```bash
python -m pip install -e ".[pretrained]"
```

PyTorch/TorchAudio 必须来自兼容版本与相同 CPU/CUDA 渠道。安装后运行：

```bash
python -c "import ser_lib; print(ser_lib.__version__)"
ser components list --json
```

若命令找不到，先确认当前 shell 已激活虚拟环境，并可用
`python -m ser_lib.cli` 代替 `ser` 排查入口安装问题。
