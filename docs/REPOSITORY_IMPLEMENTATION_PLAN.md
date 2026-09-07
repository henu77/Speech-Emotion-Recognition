# SER 基础库整体模块实施计划

> 本文件是 `Speech-Emotion-Recognition` 基础仓库的上位实施总纲。
>
> 仓库边界：本仓库只实现可复用的 SER 能力，不包含桌面端、Web UI、本地 HTTP 服务、页面状态、工作区数据库或桌面打包代码。桌面产品必须使用独立仓库，通过已发布的 Python API、CLI 或双方另行定义的适配层集成。
>
> 基线版本：`9253df8`（2026-09-07）。数据层详细契约见 [`DATA_PIPELINE_REFACTOR_PLAN.md`](./DATA_PIPELINE_REFACTOR_PLAN.md)。

## 实施进度（2026-09-07）

- W01 已完成首轮实现：严格配置基类、版本化 YAML 读取、确定性路径解析、统一根异常、结构化事件、取消令牌和库日志工具已经落地；数据异常已接入统一根异常。
- W02 已完成：建立 Windows/Linux × Python 3.10/3.12 CI、发行包构建、编译检查、pytest、完整 Pyflakes/Ruff 和全库 Mypy 门禁。
- W03 已完成：冻结 `SERModel`/`ModelOutput` 配置与输出契约，模型注册表支持独立配置校验和 descriptor 查询，CNNBaseline 增加配置 round-trip、参数统计、输入/mask 校验，并允许 artifact 从模型实例安全推导配置。
- W04 已完成：新增可配置的单向/双向 GRU 基线，使用 packed sequence 正确忽略 padding，支持 mask/length 一致性校验、训练和 artifact 恢复。
- W05 已完成首轮实现：新增版本化 `ExperimentConfig`、实验路径解析、白名单 AdamW/Adam/SGD 与 Step/Cosine 调度器、实验静态兼容预检；Trainer 支持确定性 seed、CUDA AMP 约束、梯度累积、事件、取消和 scheduler stepping。
- W06 已完成：Evaluator 提供 loss、accuracy、WAR、UAR、macro-F1、逐类 precision/recall/F1/support、混淆矩阵和样本级概率，并可原子导出 JSON/JSONL 报告。
- W07 已完成：checkpoint v2 保存并恢复模型、optimizer、scheduler、AMP scaler、随机数状态和训练配置；Trainer 可从下一 epoch 继续，且仍兼容可信的 v1 checkpoint。
- W08 已完成：artifact schema v2 默认使用 safetensors，整体原子导出，校验所有组成文件，包含输入规格、独立配置/标签/指标文件和模型卡；拒绝覆盖与路径逃逸，旧 v1 pickle 仅在显式信任时加载。
- W09 已完成：新增真实批次 forward 的文件列表、目录和 manifest 推理，支持 batch size、稳定扫描、片段记录、滑窗按 `window_map` 独立聚合、fail-fast/逐条错误隔离、取消/进度及 JSONL/CSV 原子导出。
- W10 已完成：新增可安装的 `ser` 命令入口，支持组件发现以及 dataset scan/import/validate/stats；命令复用公共 API，提供人类可读/JSON 输出和配置、数据、运行错误的稳定退出码。
- W11 已完成首轮实现：CLI 已串联 ExperimentConfig 训练、checkpoint 恢复、artifact 评估/推理/检查/校验，以及从可信 checkpoint 导出 safetensors artifact；未显式配置时 checkpoint 写入 `output_dir/checkpoints`。
- W12 已完成首轮实现：新增纯 PCM 同步流式会话、有状态线性重采样、窗口/hop、静音抑制、EMA 概率平滑、chunk 背压上限、flush/reset/close 生命周期和算法延迟报告；不同 chunk 切分产生一致窗口，缓冲保持有界。
- W13 已完成：新增轻量帧级 Transformer 基线，提供严格配置、输入投影、动态正弦位置编码、padding mask、masked mean pooling、注册表发现、训练和 artifact 恢复支持。
- W14 已完成首轮实现：新增可选 `hf_audio_classifier` 预训练语音编码器适配，默认仅本地加载并禁用远程代码；支持冻结、mask pooling、采样率预检，并将 encoder 架构配置固化进 artifact 以便脱离原模型路径恢复。
- W15 已完成首轮实现：新增严格 RAVDESS 官方文件名适配器及许可提示，不包含下载逻辑；数据 profile 可统计片段时长、采样率、声道和损坏文件；新增带环境指纹、原子 JSON 结果及回归阈值比较的可重复 benchmark 工具。
- W16 已完成第一阶段：补齐 MIT License、贡献/安全/变更/第三方许可清单及安装、数据、模型、训练 CLI、artifact/API 文档；CI 增加 macOS、wheel smoke test 和分包覆盖门禁。Notebook 经审计仅为 Markdown 提纲，已从“可执行教程”声明中降级，仍待逐篇实现。
- 当前本地验收：sdist/wheel 构建通过、完整 Ruff/Pyflakes 通过、全库 Mypy 通过、`123 passed`；分包覆盖率 Core 92.65%、Artifacts 90.76%、Engine 91.33%、Inference 89.03%、Models 87.14%、Data 66.71%、CLI 71.62%。

## 1. 项目目标与边界

本项目应成为可安装、可扩展、可测试的通用语音情感识别 Python 库：

```text
外部数据 → Manifest → 音频加载/表示/增强 → Batch
→ 模型训练 → 评估/Checkpoint → Artifact → 离线或流式推理
```

首个稳定版本必须支持：

- Windows、Linux 和 macOS 安装；
- 文件夹、CSV、JSONL 和 CASIA 数据导入；
- Waveform、Spectrogram、Mel、Log-Mel、MFCC 和基础声学特征；
- 至少一个稳定 CNN 和一个稳定时序模型；
- 分类训练、验证、测试、断点恢复和常用 SER 指标；
- 自包含、可校验的模型 artifact；
- 单文件、批量和流式推理核心；
- Python API、CLI、教程、测试、CI 和发布流程。

本仓库不负责：

- Web 页面、桌面壳和产品前端；
- FastAPI 等产品服务端和前后端通信；
- 产品工作区、用户项目数据库和任务中心；
- 麦克风设备 UI、文件选择器、安装包与自动更新；
- 云训练、账户系统、多人协作和在线标注；
- 执行任意不可信用户 Python 代码。

## 2. 当前实现快照

`完成` 表示已有测试覆盖的基础闭环；`部分完成` 表示只有最小接口；`待实现` 表示不可作为公开能力宣称。

| 模块 | 状态 | 已有能力 | 主要缺口 |
|---|---|---|---|
| `ser_lib.data` | 完成 | 统一 Dataset、manifest、加载、表示、增强、collate、缓存、导入、校验和音频属性 profile | 后续增加更多许可允许的语料适配器和超大 manifest 压测 |
| `ser_lib.models` | 完成 | 稳定模型契约、CNNBaseline、变长 GRUBaseline、轻量 Transformer、可选 HF 语音编码器适配 | 后续按需求增加受控模型族与多任务输出 |
| `ser_lib.engine` | 完成 | 实验配置、可复现 Trainer、完整分类评估、checkpoint v2 与恢复 | 后续按新模型需求增量扩展 |
| `ser_lib.artifacts` | 完成 | schema v2、safetensors、全文件校验、模型卡、原子导出及 v1 受控兼容 | 后续按新 schema 需求提供迁移工具 |
| `ser_lib.inference` | 完成 | 单文件/真实批量推理、滑窗聚合、结果导出、纯 PCM 流式会话 | 缺系统级冷启动、吞吐、峰值内存性能报告 |
| `ser_lib.core` | 完成 | 严格配置、版本检查、路径、根异常、事件、取消和日志 | 后续随业务模块扩展事件字段 |
| `ser_lib.cli` | 完成 | 组件、dataset、train、evaluate、predict 与 artifact 命令闭环，支持 JSON 输出 | 后续补 Ctrl+C 进程级集成测试和更多覆盖参数 |
| 数据处理工具 | 部分完成 | 文件夹、CSV、JSONL、CASIA、RAVDESS importer，已接入 dataset CLI | 缺恢复执行和更多适配器 |
| Benchmark | 完成 | 数据管线人工基准、环境化 JSON 结果和回归阈值比较 | 后续积累各平台正式基线数据 |
| 文档与教程 | 部分完成 | README、安装/数据/模型/CLI/artifact 文档和 0–7 课程提纲 | Notebook 尚无可执行代码；缺完整 API reference |
| 工程化 | 部分完成 | `pyproject.toml`、pytest、跨平台 CI、发行构建、完整 Pyflakes/Ruff 与全库 Mypy | 缺覆盖率分层门槛、macOS smoke test 和发布自动化 |

移除服务专属代码并完成 W01–W15 后，当前自动化测试基线为 `117 passed`，后续不得只报告新增测试而忽略全量回归。

## 3. 目标模块与依赖方向

```text
CLI / 用户 Python 代码
          │
          ├── data       数据来源、输入表示和批处理
          ├── models     模型契约、注册表和内置模型
          ├── engine     训练、评估、指标和 checkpoint
          ├── artifacts  可移植模型产物
          └── inference  离线、批量和流式推理
                         │
                         ▼
                core（配置、异常、日志、事件、版本）
```

硬性约束：

- 库代码不得依赖 FastAPI、Tauri、Electron 或前端框架；
- 领域模块不得反向依赖 CLI；
- Data 不理解模型结构，Model 不读取 Dataset；
- Engine 只依据公开契约编排，不按 representation 名称分支；
- Inference 从 artifact 恢复与训练一致的确定性预处理；
- import 阶段不得扫描、下载、初始化 CUDA 或启动线程。

## 4. 跨模块公共契约

### 4.1 配置与路径

- 公开配置使用 Pydantic，默认 `extra="forbid"`；
- 持久化配置包含 `schema_version`；
- 相对路径基于配置或 manifest 所在位置解析，不依赖 `cwd`；
- 配置不保存模型实例、设备句柄、锁或打开的文件；
- 训练开始前保存完全解析的不可变运行快照。

### 4.2 数据与模型

- `AudioRecord`、`AudioData`、`TensorSpec`、`SERSample`、`SERBatch` 是冻结边界；
- Model 通过 `ModelSpec` 声明 key、layout、feature dimension 和 batching 限制；
- Model 输出统一为 `ModelOutput`，至少包含 `logits`；
- 标签映射是 manifest 与 artifact 的显式元数据；
- 训练和推理前执行同一兼容性校验。

### 4.3 事件、错误与取消

- 长操作通过 Python callback/protocol 上报进度、指标和日志事件；
- 事件协议不得包含 HTTP、页面或数据库概念；
- 领域异常保留 UID、路径、组件、阶段和原始 `__cause__`；
- 批量操作明确 `fail_fast` 或“记录错误继续”策略；
- 取消使用可选 token/callback，并保证产物原子性。

### 4.4 安全

- 默认不加载不可信 pickle，权重优先使用 `safetensors`；
- 解压和导入防止路径穿越与意外覆盖；
- 外部模型通过受控 adapter/plugin 契约接入；
- 日志不输出密钥、完整环境变量或原始音频内容。

## 5. 模块实施计划

### 5.1 Core

- 实现严格基础配置、版本化读写和路径工具；
- 整理核心异常与领域异常的继承边界；
- 定义结构化事件和取消协议；
- 提供库友好的 logging 配置，默认不添加 handler；
- 暴露版本信息和清晰的可选依赖错误。

验收：Core 无重型依赖；配置 round-trip、版本错误、Unicode 路径和事件序列有测试。

### 5.2 Data

统一数据流水线是架构基线，禁止重新加入 Waveform/MFCC/Mel 专属 Dataset。

- 完善公开 API、shape 文档和自定义组件示例；
- Importer 增加 dry-run、冲突策略、原子写入、进度和取消；
- 增加类别、时长、采样率、声道和损坏文件统计；
- 固化 train/eval/inference transform profile，后两者默认确定性；
- 增加 Unicode 路径、多 worker、长/极短音频和大 manifest 测试；
- Cache 增加容量、清理、版本和并发压力测试；
- 数据处理脚本调用公开 importer 或 CLI，不复制实现。

验收：Dataset 与推理对同一输入的确定性预处理一致；错误可定位；benchmark 可重复。

### 5.3 Models

1. 冻结 `SERModel`、`ModelOutput`、`ModelSpec` 和 registry；
2. 完善 CNNBaseline 的输入校验、配置 round-trip、参数统计和导出；
3. 实现 BiLSTM/GRU 基线，正确处理 mask/length；
4. 实现 CNN-RNN 或轻量 Transformer；
5. 以可选依赖加入一种预训练编码器适配；
6. 定义受控第三方模型 adapter；
7. 分类链路稳定后再扩展 valence/arousal 多任务输出。

每个模型必须提供严格配置、descriptor、ModelSpec、forward、变长 batch、训练一步和 artifact round-trip 测试。

### 5.4 Engine

- 定义 `ExperimentConfig`，组合数据、模型、优化器、调度器和输出；
- 实现 seed、设备策略、AMP、梯度裁剪和梯度累积；
- 优化器和调度器使用注册表或白名单，禁止 `eval`；
- 增加 early stopping、best/last checkpoint 和完整恢复；
- 指标至少包括 loss、accuracy、UAR、WAR、macro-F1 和 confusion matrix；
- 类别不平衡支持 class weight 与 sampler，并记录策略；
- Trainer 通过 callback 产生事件并支持安全取消；
- 历史保存 JSONL，评估输出样本预测和聚合报告；
- 首版保证 CPU 与单 GPU，DDP 后置。

验收：固定 seed 的小训练可复现；中断恢复与连续训练在容差内一致；取消不生成伪成功 artifact。

### 5.5 Artifacts

Checkpoint 用于继续训练，artifact 用于分发推理，两者不得混用。

```text
model-artifact/
├── manifest.json
├── weights.safetensors
├── data_config.json
├── model_config.json
├── labels.json
├── metrics.json
└── README.md
```

- Manifest 包含 schema、库版本、模型类型、输入规格、标签、摘要、格式和校验和；
- 添加 safetensors；保留 `.pt` 时明确可信来源限制；
- 实现原子导出、完整性验证、版本检查和迁移提示；
- 增加数据来源、许可、语言、限制和指标等模型卡字段；
- ONNX 独立排期，不阻塞 Python v1。

验收：干净环境仅凭 artifact 可恢复并推理；篡改文件会被拒绝。

### 5.6 Inference

1. 完善单文件推理的设备、阈值、top-k、滑窗和错误上下文；
2. 增加文件列表、目录和 manifest 批量推理；
3. 输出 JSONL/CSV，支持片段概率、聚合概率和可选 embedding；
4. 建立冷启动、延迟、吞吐和峰值内存 benchmark；
5. 实现纯 PCM 流式核心：环形缓冲、窗口、步长、状态重采样、静音策略和平滑；
6. 流式核心不枚举麦克风，也不控制任何界面；
7. 定义 backpressure 和会话释放，防止无限缓存。

流式验收：不同 chunk 分割产生一致窗口；长时间运行内存稳定；会话可重置和停止；有算法延迟报告。

### 5.7 CLI

```text
ser components list
ser dataset scan|import|validate|stats
ser train
ser evaluate
ser predict
ser artifact inspect|verify|export
```

- CLI 调用公共 API，不复制业务逻辑；
- 支持配置文件和少量显式覆盖；
- 默认人类可读，`--json` 适合脚本；
- 退出码区分配置、数据、兼容性和运行错误；
- 测试 Windows 路径、空格、Unicode 和 Ctrl+C。

### 5.8 数据集适配工具

- 将根目录 `data/*.py` 逐步迁移为 importer 或 CLI；
- 不下载受限数据集，只提供目录和获取说明；
- 每个适配器明确许可、目录规则、标签映射和划分；
- 默认引用音频，复制必须显式；
- 提供 dry-run 和可复现转换摘要。

### 5.9 文档、教程与发布

- README 只展示已验证接口；
- 建立安装、数据格式、模型开发、训练、artifact、CLI 和排错文档；
- Notebook 纳入执行检查并移除过时 API；
- 教程使用可再分发或合成 fixture；
- 增加贡献指南、安全说明、changelog 和许可证清单；
- 使用语义化版本，公开后变更接口必须经过弃用期。

### 5.10 测试与质量

- 单元测试覆盖配置、契约、表示、模型 forward 和指标；
- 集成测试覆盖 importer → data → train → artifact → inference；
- 使用 Ruff、类型检查和依赖安全扫描；
- wheel/sdist 在干净环境安装并 smoke test；
- CI 覆盖最低/主 Python 版本、Windows/Linux 和 CPU；
- GPU 与大型模型使用可选定时任务；
- 覆盖率按核心模块设门槛，不用单一数字掩盖关键链路缺测。

## 6. 实施里程碑

### M0：工程基线与契约冻结

Core、CI、lint、类型检查、构建测试和公共 API 清单完成。

### M1：可复现训练闭环

CNN 完善、时序基线、ExperimentConfig、指标、恢复训练和 CLI train/evaluate 完成。

### M2：可分发模型与批量推理

Artifact v2、安全权重、模型卡、批量推理和 CLI artifact/predict 完成。

### M3：流式推理与模型扩展

纯 PCM 流式核心、轻量 Transformer 和首个预训练适配完成，并有性能报告。

### M4：首个稳定开源发布

跨平台 CI、文档、教程执行、许可审查、构建发布和兼容测试全部通过。

## 7. 编码智能体工作包

| 编号 | 工作包 | 前置 | 输出 |
|---|---|---|---|
| W01 | Core 配置、事件、日志和异常 | 无 | 核心协议与测试 |
| W02 | CI、Ruff、类型检查、构建 smoke test | 无 | 工程门禁 |
| W03 | 模型协议与 CNN 完善 | W01 | 稳定模型 API |
| W04 | GRU/BiLSTM 时序基线 | W03 | 时序模型与测试 |
| W05 | ExperimentConfig 与 Trainer 增强 | W01、W03 | 可复现训练 |
| W06 | 指标、报告和样本预测 | W05 | 完整评估 |
| W07 | Checkpoint 恢复与安全取消 | W05 | 恢复测试 |
| W08 | Artifact v2 与 safetensors | W03、W05 | 分发格式 |
| W09 | 批量离线推理 | W06、W08 | JSONL/CSV 输出 |
| W10 | CLI 框架与 dataset 命令 | W01 | `ser` 入口 |
| W11 | CLI train/evaluate/predict/artifact | W05、W08–W10 | 命令行闭环 |
| W12 | 流式推理核心 | W08、W09 | PCM 会话 |
| W13 | 轻量 Transformer | W03–W05 | 第二类模型 |
| W14 | 预训练编码器适配 | W03、W08 | 可选模型族 |
| W15 | 数据适配器与 benchmark | W02 | 数据扩展报告 |
| W16 | 文档、教程、发布 QA | 持续 | 稳定发布 |

执行规则：

1. 开始前阅读本文件、模块源码和测试；
2. 明确允许修改的模块、公共接口和验收命令；
3. 公共契约变化先更新设计文档；
4. 不重新引入旧 Dataset 或 tuple batch；
5. 不用 `except Exception: pass` 隐藏主错误；
6. 新组件必须可注册、枚举、校验并从公共入口导出；
7. 新依赖说明用途、可选性、许可证、平台影响和体积；
8. 运行相关测试、全量测试和静态检查；
9. 不提交真实语音、模型大权重、缓存、密钥或机器路径；
10. 不创建 UI、HTTP 服务或桌面产品代码。

## 8. 仓库级集成验收

```text
数据链路：scan → manifest → validate → DataConfig → Batch
训练链路：兼容检查 → train → best/last checkpoint → evaluate
分发链路：checkpoint + configs + labels → artifact → 新进程推理
流式链路：相同 PCM 的不同 chunk 分割 → 相同窗口与稳定聚合
失败链路：损坏数据/磁盘不足/中断 → 明确异常 → 原子清理 → 可重试
```

## 9. Definition of Done

模块完成必须同时满足：

- 公共行为、输入输出和失败模式有文档；
- 配置严格校验且版本策略明确；
- 正常、边界和错误路径有测试；
- 与上下游至少有一个集成测试；
- 无 import-time 副作用；
- 不依赖当前目录或开发者机器路径；
- 日志、事件、错误和取消遵守统一契约；
- 从公共入口可用；
- 全量测试和质量门禁通过；
- README 不宣称尚未实现的功能；
- 不包含桌面端、Web UI 或产品服务代码。

## 10. 文档维护与架构决策

Artifact 格式、公共契约、自定义代码边界、支持平台、最低依赖版本、大型依赖以及跨仓库集成协议发生变化时，必须新增 ADR 或更新本文件。

每完成一个工作包，更新第 2 节状态和对应里程碑，记录测试结果或关键性能基线。计划记录目标与验收，不写成逐行开发日志。

---

总原则：这个仓库只提供可靠、可组合、与界面无关的 SER 能力。任何桌面产品、Web 页面和产品级本地服务都在独立仓库建设。
