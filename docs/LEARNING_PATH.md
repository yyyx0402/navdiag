# 使用成品，再系统学习 NavDiag

适合已经会 Python、刚开始学机器学习的使用者。先读取当前应用和已保存结果，每次用30–60分钟理解一个问题、读一个函数、完成一个小练习。推进速度由理解程度决定，无需先重写整个项目。

文档更新于2026-10-12。当前主线是卫星 LOS/NLOS；定位告警是另一种标签任务；RAG 负责检索与解释已有证据。[README](../README.md)提供启动步骤，[路线图](ROADMAP.md)区分已实现和待研究内容。

## 先看成品和三类数据

本机已具备产物时，在项目根目录读取现有应用：

```powershell
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/urbannav_positioning --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization
```

依次查看卫星识别详情、区域适应性、定位覆盖与断解、诊断助手和文献学习。新克隆的仓库尚无数据/模型时，先按 README 的 `artifacts/learning_demo` 模拟入口体验，再阅读真实实验公开汇总。

| 数据粒度          | 先看什么                                  | 禁止混淆的概念                                   |
| ----------------- | ----------------------------------------- | ------------------------------------------------ |
| 时间 × 卫星       | KLT/UrbanNav 的 `signal_label` 和五项特征 | LOS/NLOS不是建筑/树木类别；LOS也可能含多径       |
| 接收机历元        | UrbanNav 单点解、参考轨迹和水平误差       | `>10 m` 是告警正类，不能当作NLOS标签或定位精度   |
| 文档段落/研究摘要 | 来源、章节、阅读范围和 `[E#]`             | 引用ID存在不等于结论受支持；论文结果不是项目成绩 |

学习改动放在 `artifacts/learning/` 或独立副本中，保留主实验和测试集。默认练习采用手算、合成小表、已有测试与只读回放，不恢复 NLOS 调参。

## 分阶段路线

### 1. 信号、特征与标签

先读[第一课](lessons/01_features_labels_splits.md)、[NLOS与论文概念](lessons/05_nlos_zhang2023.md)和[标签协议](OBSTRUCTION_LABELS.md)。源码从 [nlos.py](../src/navdiag/nlos.py) 的 `NLOS_FEATURES`、`validate_observations` 开始，再比较 [data.py](../src/navdiag/data.py) 的 `FEATURES`。

需要掌握：一个样本是什么、X/y是什么、参考来源、unknown、缺失值和数据泄漏。暂时不要求记住全部GNSS误差公式。

练习：从公开说明或已有表中选一条卫星观测，逐列标注“输入 / 标签 / 分组 / 参考审计”；再写出一条位置历元的分类标签。验收：能解释为什么标签、参考位置和区域ID不能进入学习特征，以及缺少合格参考不意味着“正常”。

### 2. LR/RF 与误报、漏报

读[第三课](lessons/03_models.md)。源码顺序为 [nlos.py](../src/navdiag/nlos.py) 的 `build_classic_model` → [learning.py](../src/navdiag/learning.py) 的 `evaluate`、`choose_threshold` → 已有报告的 `models`。

需要掌握：`fit`/`predict_proba`、Pipeline、填充与标准化、类别不均衡、过拟合，以及Precision/Recall/F1。NLOS RF默认200树/深度12/最小叶8；告警 RF默认150树/深度8/最小叶6，它们是两个配置。

练习：手算 `TP=8, FP=2, FN=4, TN=6`，再用十个合成分数比较0.3和0.7阈值。参数练习只在小型合成训练/验证数据进行。验收：能解释为什么概率评分不等于已校准概率、阈值来自验证集，以及没有正样本时Recall不能评估。

### 3. 整段划分和区域评估

读[第六课](lessons/06_generalization.md)、[KLT配对协议](experiments/nlos_protocols.md)与[UrbanNav结果](experiments/urbannav_generalization.md)。代码看 [nlos_protocols.py](../src/navdiag/nlos_protocols.py) 的 `protocol_manifests`、`summarize_protocols`，再看 [nlos_external.py](../src/navdiag/nlos_external.py) 的 `area_manifests`、`external_evaluation`；对应设置在 [KLT计划](../config/nlos_protocols.json)与[UrbanNav计划](../config/nlos_generalization.json)。

需要掌握：训练/验证/测试职责、同域与目标区域排除、冻结迁移、逐区域宏平均、采集独立性和参考覆盖。6∶2∶2是整段划分的目标，不通过打散相邻历元强凑。

练习：只读一对协议，列出共同验证/测试段和移除的训练段，核对无交叉；手算三个区域F1平均。验收：能解释固定C/N0规则为何两列相同，为什么同城不能当作跨城市，以及减少训练量也是跨区域差值中的影响因素。

### 4. 五种神经网络与因果历史

代码从 [nlos_transformer.py](../src/navdiag/nlos_transformer.py) 的 `satellite_windows` 开始，再看 [nlos_neural.py](../src/navdiag/nlos_neural.py) 的 `build_network`、`train_neural`。阅读顺序建议 MLP → CNN → GRU/LSTM → Transformer；数学按需要补矩阵运算、梯度下降、BCE和注意力。

五网络输入为同一卫星过去及当前32次、每次五特征的观测；LR/RF读取当前特征，历史访问不同。当前Transformer未包含Zhang论文的跨卫星联合结构，见[复现核对](research/zhang2023_reproduction.md)。

练习：画出前三次观测的短历史补齐、遇到间隙时重置；阅读 [test_nlos_transformer.py](../tests/test_nlos_transformer.py) 的因果检查，再在小副本中加入一个未来观测。验收：能解释窗口张量、padding、早停和推理历史；不能仅凭一次测试F1将差异归因于架构。

### 5. 从伪距到位置，再到滤波

读[第二课](lessons/02_positioning.md)和新增[定位与覆盖课](lessons/08_positioning_coverage.md)。源码顺序： [positioning.py](../src/navdiag/positioning.py) 的 `solve_wls`、`enu_rotation` → [fusion.py](../src/navdiag/fusion.py) 的 `filter_positions` → [doppler.py](../src/navdiag/doppler.py) 的 `filter_doppler_positions`。

再看 [prepare_positioning.py](../tools/urbannav/prepare_positioning.py)、[prepare_velocity.py](../tools/urbannav/prepare_velocity.py)、[context.py](../src/navdiag/context.py)和[availability.py](../src/navdiag/availability.py)，理解参考评分与完整历史如何分开。原始工具使用独立GPL许可。

练习：在合成伪距例子中加30 m偏差；手算旺角139/3350、577/3350、139/1270分别代表什么。已有产物可用 `scripts/review_positioning.py` 只读回放。验收：能解释ECEF/ENU、位置与速度状态、Q/R/NIS、Doppler来源，以及为什么RMSE改善可能伴随P95退步。原GSDC 11点是历史解析样例，不再是当前唯一真实测试。

### 6. RAG、证据问答与自选模型

先读[第四课](lessons/04_agent.md)和新增[RAG学习课](lessons/07_rag.md)。源码顺序： [knowledge.py](../src/navdiag/knowledge.py) 的 `retrieve_knowledge` → [semantic.py](../src/navdiag/semantic.py) 的 `similarity` → [literature.py](../src/navdiag/literature.py) 的 `reading_list` → [agent.py](../src/navdiag/agent.py) 的 `EvidenceTools.invoke`、`ask` → [agent_context.py](../src/navdiag/agent_context.py) 的 `evidence_context`。

需要掌握：文档与chunk、关键词/语义检索、top-k、相关性门槛、来源引用、上下文预算；并区分离线模板、一次证据问答和多轮工具Agent。BGE是本地编码器，GLM等是可自选的回答模型；应用没有自动联网搜索。

练习：对一个术语、一条数值问题和一个无关问题记录命中来源、工具和引用；用网页文献学习找一篇论文，核对“已阅读范围”和“本项目关系”。先采用 `--provider offline`，实际API设置见[接口说明](LLM_API.md)。验收：能跟踪“问题→检索/工具→依据→答案”，并指出一个引用存在但结论不受支持的例子。

### 7. 验证、复现与项目表达

读[项目检查](PROJECT_REVIEW.md)、[真实GLM检查](experiments/glm_quality.md)与[定位方法出处](research/positioning_methods.md)。查看 [tests](../tests)、[review_project.py](../scripts/review_project.py)、[review_positioning.py](../scripts/review_positioning.py)与[固定计划](../config/urbannav_positioning.json)。

练习：选择一个参考隔离/因果测试，解释其输入、断言、能证明和不能证明的内容；再核对一条公开汇总对应的数据分母与SHA。API回答审查应单独检查数字、引用支持、过度归因和可读性。

验收：用五分钟讲清专业问题、数据与标签、划分、基线与网络、主要失败、Doppler方法参考和RAG验证。求职描述只使用自己已经实际理解的内容，明确同城研究、参考标签和未复现的算法。

## 最小学习验收清单

- 能分别解释NLOS五项特征、告警六项特征和两种正类。
- 能手算混淆矩阵、验证阈值与逐区域F1平均，并找出一处潜在泄漏。
- 能解释32次卫星历史、10历元告警GRU历史和时间间隙重置。
- 能同时报告定位误差与可用率，理解参考质量不能改变观测历史。
- 能从检索片段和实验工具找到依据，区分引用检查与事实审查。
- 能说明当前尚缺跨城市验证、同源NLOS定位闭环、TDCP/FGO和完整原始IMU处理。

可在自己的笔记中固定记录四件事：这个函数解决什么问题、输入/输出是什么、怎样验证、还不能推出什么结论。遇到不懂的数学或一段代码，再针对该处提问；无需同时补完全部理论。
