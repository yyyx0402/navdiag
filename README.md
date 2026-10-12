# NavDiag

[![Python checks](https://github.com/yyyx0402/navdiag/actions/workflows/ci.yml/badge.svg)](https://github.com/yyyx0402/navdiag/actions/workflows/ci.yml)

**GNSS 信号识别、定位分析与有来源的诊断问答。**

NavDiag 是一个本地 GNSS 分析应用，集成逐卫星 LOS/NLOS 识别、区域适应性对比、定位与告警可视化、RAG 问答和文献推荐。查看结果与查找文献无需模型 API；问答可连接自选的 Chat Completions 兼容服务。

![NLOS 工作台](docs/images/nlos-workbench.png)

## 快速运行

推荐 Python 3.12。新克隆的仓库不包含数据或模型，以下命令生成模拟示例：

```powershell
git clone https://github.com/yyyx0402/navdiag.git
cd navdiag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m navdiag demo --navigation --out artifacts/demo
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/demo
```

打开 [本地应用](http://127.0.0.1:8765)。模拟结果仅用于体验流程。Linux/macOS 使用 `.venv/bin/python`。

已有真实实验产物时，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/urbannav_positioning --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization
```

五网络、RINEX 解算和语义检索分别需要 `temporal`、`gnss`、`rag` 可选依赖，安装及复现见[文档](docs/README.md)。模型连接在网页“诊断助手 → 自定义模型 API”中配置，详见[接口说明](docs/LLM_API.md)。

## 功能与结果

| 功能 | 内容 | 说明 |
| --- | --- | --- |
| 卫星信号识别 | LR、RF、C/N0 规则，以及 MLP/CNN/LSTM/GRU/Transformer | [标签与模型](docs/MODEL_CARD.md) |
| 区域适应性 | KLT 同域/跨区域配对，冻结迁移，UrbanNav 城区互斥 | [KLT](docs/experiments/nlos_protocols.md)、[UrbanNav](docs/experiments/urbannav_generalization.md) |
| 定位与告警 | SPP、固定/自适应/Doppler KF，整段留出、可用率和断解分析 | [定位结果](docs/experiments/positioning_expansion.md) |
| 诊断助手 | 本地检索、只读指标工具、来源引用、自选模型 | [RAG](docs/RAG_RETRIEVAL.md) |
| 文献推荐 | 20 篇研究摘要，按问题、主题与难度筛选 | [文献目录](docs/research/reading_list.md) |

当前 UrbanNav 外部迁移/城区互斥平均 NLOS F1：RF 为 **0.7884/0.7852**，Transformer 为 **0.7852/0.7489**；逐城区排名存在差异。全部区域均在香港，标签覆盖不完整，结果不能代表跨城市泛化。

三条约 94.5 分钟路线共有 1193 个可评分定位点。旺角 SPP/Doppler KF 的水平 RMSE 为 **15.481/14.610 m**，但 Doppler 的 Median/P95 变差；SPP 全段可用率仅 **17.2%**。当前是离线分析原型，尚未达到连续可靠导航效果。告警定义为基线水平误差 >10 m，与卫星 NLOS 是两个任务；10 m 不是定位精度目标。

## 开发与许可

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_release.py
```

源码在 [src/navdiag](src/navdiag)，固定计划在 [config](config)，适配器在 [tools](tools)。数据、权重、缓存与连接密钥保留在本机；公开仓库提供代码、来源清单和当前汇总报告。原创主包采用 [MIT](LICENSE)，独立数据提取工具采用各自 GPL 许可，详见[复用记录](docs/OPEN_SOURCE_REFERENCES.md)。贡献说明见 [CONTRIBUTING](CONTRIBUTING.md)。
