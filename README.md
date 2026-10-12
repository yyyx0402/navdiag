# NavDiag

[![Python checks](https://github.com/yyyx0402/navdiag/actions/workflows/ci.yml/badge.svg)](https://github.com/yyyx0402/navdiag/actions/workflows/ci.yml)

**GNSS 信号识别、定位分析与有来源的诊断问答**

NavDiag 是结合卫星导航、机器学习与 RAG 的本地应用。它提供逐卫星 LOS/NLOS 识别、区域适应性对比、定位与失准告警分析，以及按问题查找文献的学习入口。适合学习、离线研究与求职项目展示。

当前版本为 v0.3，文档更新于 **2026-10-12**。查看结果、使用离线诊断和查找文献无需模型 API；问答支持自行配置 Chat Completions 兼容服务。

## 可以做什么

| 功能         | 已实现内容                                                                                        | 阅读入口                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 卫星信号识别 | LR、RF、C/N0 规则与 MLP、CNN、LSTM、GRU、Transformer 五网络；混淆矩阵、阈值权衡与逐卫星检查       | [标签说明](docs/OBSTRUCTION_LABELS.md)、[模型卡](docs/MODEL_CARD.md)                                        |
| 区域适应性   | KLT 同域/跨区域配对实验；冻结 KLT 模型迁移至 UrbanNav；UrbanNav 六个城区互斥方向                  | [双协议结果](docs/experiments/nlos_protocols.md)、[新城区结果](docs/experiments/urbannav_generalization.md) |
| 定位与告警   | RTKLIB SPP、固定/风险自适应 KF、Doppler 速度约束 KF；三条整段路线轮流留出；原始可用率和断解时间线 | [完整路线评估](docs/experiments/positioning_expansion.md)、[方法出处](docs/research/positioning_methods.md) |
| 诊断问答     | 只读实验工具、本地语义/关键词检索、来源行号与引用检查；离线速查、证据问答和工具 Agent             | [RAG](docs/RAG_RETRIEVAL.md)、[模型与 API](docs/LLM_API.md)                                                 |
| 文献学习     | 20 篇有出处和阅读范围的研究摘要；按问题、主题和难度筛选，提供原文入口和阅读提示                   | [文献目录](docs/research/reading_list.md)、[学习路线](docs/LEARNING_PATH.md)                                |

各面板和精确指标表可展开、收起；模型对比支持切换 Precision / Recall / F1。文献推荐在本地执行，不调用问答模型。当前应用的 LLM 没有联网搜索工具。

![已保存的 NLOS 工作台截图；定位和文献功能以当前应用为准](docs/images/nlos-workbench.png)

## 两种任务，分别评估

| 任务          | 一条样本       | 正类与参考                                         | 学习输入                                                  |
| ------------- | -------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 卫星 LOS/NLOS | 时间 × 卫星    | NLOS；KLT 相机天空掩膜或 UrbanNav 静态建筑几何参考 | 五项无线电/几何特征；神经网络使用同一卫星的过去及当前观测 |
| 定位失准告警  | 接收机定位历元 | 基线水平误差 >10 m；独立参考轨迹生成标签           | 卫星数、C/N0 均值/最小值、HDOP、残差 RMS、因果估计速度    |

参考误差、参考位置和标签不作为模型输入。`unknown` 不参与 NLOS 监督或评分，但其可用观测可以保留在因果历史中。10 m 是告警定义，位置精度另用 Median/P95/RMSE 和误差内比例评估。当前位置告警和滤波尚未完成“逐卫星 NLOS 预测 → 观测加权 → 定位收益”的闭环。

## 运行应用

### 已有本地产物

在项目根目录执行以下命令读取已完成结果，不重新训练：

```powershell
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/urbannav_positioning --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization
```

打开 **http://127.0.0.1:8765**。定位面板读取旺角主实验及三路线对照；卫星识别读取 KLT 与 UrbanNav 的独立报告。VS Code 的 `NavDiag: dashboard` 入口使用同样的目录。原始数据、模型权重和实验产物由 Git 忽略，新克隆的仓库需要先生成它们。

### 从源码快速体验

推荐 Python 3.12。以下入口生成独立的模拟学习产物，运行 LR/RF 和定位流程；真实识别数据的复现入口见下一节。

```powershell
git clone https://github.com/yyyx0402/navdiag.git
cd navdiag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m navdiag demo --navigation --out artifacts/learning_demo
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/learning_demo --port 8766
```

打开 **http://127.0.0.1:8766**。模拟页面仅供学习，不代表真实城市成绩。Linux/macOS 将解释器路径换为 `.venv/bin/python`。首次安装需要网络；`requirements-lock.txt` 是测试环境版本快照，PyTorch 单独安装。

需要五网络/GRU、原始 RINEX 或本地语义检索时再安装相应依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[temporal,gnss,rag]"
.\.venv\Scripts\python.exe scripts/prepare_semantic_index.py --download
```

最后一条下载固定版本的公开 BGE 权重并在本机编码，不调用模型 API。未准备语义模型时，应用明确回退到关键词检索。

## 真实数据与复现入口

| 实验               | 数据规模与用途                                                                  | 操作说明                                                                                  |
| ------------------ | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| KLT 双协议         | 九段、三个日期；2851 历元、45228 条卫星观测；六个配对实验                       | [预处理工具](tools/klt/README.md)、[固定计划与复现](docs/experiments/nlos_protocols.md)   |
| UrbanNav NLOS      | 三城区；5671 原始历元、77744 条卫星观测；27839 条通过几何参考保护，其余 unknown | [工具](tools/urbannav/README.md)、[泛化实验](docs/experiments/urbannav_generalization.md) |
| UrbanNav 定位/告警 | 三段约94.5分钟；5671 原始历元、1193 个可评分点；三路线分别留出一次              | [完整步骤与分母](docs/experiments/positioning_expansion.md)                               |
| GSDC 历史解析样例  | 修正后4572个匹配历元；完整两路线用于训练/验证，11点留出仅作解析检查             | [字段与适配](data/README.md)、[纠错记录](docs/experiments/gsdc2021_correction.md)         |

下载器记录固定 URL、版本、大小和 SHA-256，第三方数据不随仓库分发。主 Python 包采用 MIT；调用上游 GPL 提取器的独立预处理工具采用各自 GPL 许可，不包含在 wheel 中。

当前学习先读取保存的报告。复现训练应使用新产物目录，保留旧报告和冻结计划；不删除指纹检查以继续写入旧实验。历史调参保留为开发记录，当前 NLOS 主实验使用固定默认参数。详细许可、字段和时间处理见[数据说明](data/README.md)与[复用记录](docs/OPEN_SOURCE_REFERENCES.md)。

## 当前结果与实际边界

KLT 同域/跨区域使用相同的配对测试段；固定 C/N0 规则两侧成绩相同是设计所致，不能证明没有域偏移。三目标区域等权平均、逐区域成绩和验证选模依据见[协议报告](docs/experiments/nlos_protocols.md)。UrbanNav RF 的外部迁移/城区互斥平均 F1 为0.7884/0.7852，Transformer 为0.7852/0.7489；旺角的局部排名不同，不能推广为普遍模型优势。

定位的同轮水平 RMSE（米）：

| 测试路线       | RTKLIB SPP | 固定 KF | 风险自适应 KF | Doppler KF |
| -------------- | ---------: | ------: | ------------: | ---------: |
| 尖沙咀，311 点 |      7.881 |   9.559 |         6.902 |      8.531 |
| 黄埔，743 点   |     12.153 |  15.848 |        12.285 |     11.170 |
| 旺角，139 点   |     15.481 |  15.669 |        15.751 |     14.610 |

这些误差只针对有有效解且参考合格的点。旺角全段 SPP 可用率17.2%，最长断解约464秒；Doppler 虽降低 RMSE，Median/P95 却变差。完整可用率、告警误报/漏报和方法对照见[定位报告](docs/experiments/positioning_expansion.md)。当前属于离线原型，尚未支持连续可靠导航；FGO、TDCP、完整 GNSS/IMU 和 Zhang 2023 论文模型尚未复现。

所有当前区域实验均在香港，部分测试段已在开发中查看。设备、日期、参考方法和覆盖率会影响比较；跨城市与稳定模型排名仍需新的独立数据。旧 GSDC 验证 F1 0.991 已撤回。历史四段 NLOS、模拟定位和调参结果分别保存在[五网络记录](docs/experiments/five_neural_nlos.md)、[v0.2](docs/experiments/v0_2.md)与[历史搜索](docs/experiments/nlos_hyperparameter_search.md)，不混入当前结果。

## RAG、问答与文献学习

RAG 使用11条短笔记、10份项目文档和20篇原创研究摘要，按相关性取少量片段，并用只读工具读取当前数值。学习卡的 WHY/HOW/WHAT、项目关系和原文链接供使用者阅读，不把全部文献或学习指南发送给模型。文档更新后的段落数和检索检查见 [RAG 记录](docs/RAG_RETRIEVAL.md)。

网页 **诊断助手 → 自定义模型 API → 配置模型与 API** 可保存多个自选服务。证据问答先在本地准备依据、再进行一次模型请求；工具 Agent 由模型选择工具和改写检索词。连接与密钥保存在本机 Git 忽略目录，读取接口不回传密钥；支持的模型与地址由所选服务决定。

已使用用户配置的 GLM-5.3 完成17次真实证据问答，保留错误和复测。39/39离线检查验证路由、检索和引用关系；引用存在、检索命中或 API 成功都不等于回答事实正确。详情见[真实回答检查](docs/experiments/glm_quality.md)、[Agent](docs/AGENT.md)和[接口说明](docs/LLM_API.md)。

可先运行零模型 API 的本地诊断：

```powershell
.\.venv\Scripts\python.exe -m navdiag ask "比较模型的 F1" --provider offline --artifacts artifacts/urbannav_positioning --task-context satellite_nlos --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization
.\.venv\Scripts\python.exe -m navdiag ask "比较定位精度" --provider offline --artifacts artifacts/urbannav_positioning --task-context position_error
```

## 项目导航与学习

| 路径                                           | 作用                                                        |
| ---------------------------------------------- | ----------------------------------------------------------- |
| [src/navdiag](src/navdiag)                     | 数据校验、识别模型、定位/滤波、检索、证据工具与本地网页服务 |
| [config](config)                               | 数据来源、整段划分与冻结实验设置                            |
| [tools](tools) / [scripts](scripts)            | 独立原始数据适配、下载、评测和只读回放                      |
| [tests](tests)                                 | 因果历史、参考隔离、指标、API、检索和产物关系检查           |
| [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md) | 面向会Python、机器学习入门者的分阶段学习与验收任务          |
| [docs/ROADMAP.md](docs/ROADMAP.md)             | 已实现能力与后续研究缺口                                    |

先使用已有应用，再按[学习路线](docs/LEARNING_PATH.md)从 NLOS 标签和 LR/RF 开始，逐步进入区域评估、五网络、定位/Doppler 和 RAG。每阶段都有可点击源码入口、一个小练习和能否讲清的检查点。

## 验证与开源

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_release.py
.\.venv\Scripts\python.exe scripts/review_project.py
.\.venv\Scripts\python.exe scripts/review_positioning.py
```

后两项需要已完成的本地真实产物，只检查/回放，不训练。检查数、检索命中和真实生成审查的含义分别记录在[项目检查](docs/PROJECT_REVIEW.md)；远程 CI 可由顶部徽章查看，它检查代码与模拟流程，不下载完整真实数据重跑。

每次实验保留数据指纹、划分、阈值、指标和冻结设置；可公开汇总记录，原始轨迹、模型权重、环境、缓存及密钥保留在 Git 忽略目录。原创代码使用 [MIT](LICENSE)。贡献与发布见 [CONTRIBUTING](CONTRIBUTING.md) 和[发布说明](docs/PUBLISHING.md)。
