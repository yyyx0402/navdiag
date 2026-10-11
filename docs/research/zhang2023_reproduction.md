## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-11
- Verification Status: ANALYZED
- Version Label: zhang2023_validation_v1

# Zhang 2023：方法核对与复现条件

结论：可以独立实现论文的核心结构并在补齐输入后开展方法复现；当前不能声称完成作者结果复现。用户提供的八页 PDF 已阅读，特征公式、Fig. 3 结构图、Tables I–V 和实验讨论已视觉核对。PDF 结构预检 PASS（仅结构，不代表学术正确性），SHA-256 为 15cf6d88baf4d21e1e8cc9e40d2d52b684eeee60e62470f468209215c9ddbccd。

来源：[论文 arXiv v2](https://arxiv.org/abs/2309.00480v2)，Haoming Zhang、Zhanxin Wang、Heike Vallery，*Learning-based NLOS Detection and Uncertainty Prediction of GNSS Observations with Transformer-Enhanced LSTM Network*。学习解读见[第五课](../lessons/05_nlos_zhang2023.md)。原 PDF 与提取全文不复制到发布仓库。

## 作者代码与数据可用性

2026-10-11 核查作者链接 [DeepNLOSDetection](https://github.com/rwth-irt/DeepNLOSDetection)：仓库首页只有 README 和 materials，materials 中是轨迹图；README 仍显示拟发布代码/数据的信息。没有找到可运行网络、预处理、训练命令、权重或 Aachen 数据下载。结论是“当前链接未提供这些复现材料”，不能据此断言作者从未公开其他版本。没有向作者发送消息。

可用的相关开源资料：[KLT 数据集](https://github.com/ebhrz/KLTDataset)提供数据入口和解析演示；[Fossett Laboratory 教学项目](https://github.com/fossettlab/gnss-nlos-notebook)提供五项特征提取与 RF notebook；[RTDL 作者实现](https://github.com/yandex-research/rtdl-revisiting-models)提供表格模型和预处理参考。后者的 FT-Transformer 对特征列建 token，与本论文跨卫星注意力不是同一种方法。

## 论文实际做了什么

输入形状为 (B,N,T,5)：批量 B、最多 N 颗卫星、时间窗口 T、每星五特征。每颗卫星用两层 LSTM 编码历史，再由目标卫星生成 query、其他卫星上下文生成 key/value，进行多头注意力。随后 BiLSTM 融合注意力表示，和目标卫星的 MLP 特征拼接，分别输出可见性与伪距误差。MLP 为三个隐藏层、ReLU。

论文训练 DL 使用 Adam、多阶段学习率调度、分类 MSE、伪距误差回归 L1。SVM 使用 RBF、C=1；XGBoost 写明 200 树、学习率 1.0。论文没有在正文给出足以完全重建的窗口长度、LSTM 宽度、注意力头数、完整调度、归一化/填充值、标签残差阈值、精确划分和随机种子；不得按经验补值后声称原样复现。

Fig. 3 与文字不足以确定 BiLSTM 的完整张量轴、初始化映射和卫星缺失处理。“detach”也不能仅凭文字确定为 PyTorch 的停止梯度操作。独立实现必须写明这些选择。双向处理若只在当前时刻已收集到的历史/上下文内进行，不必读取未来新观测；是否可在线使用要按实际时间索引验证。

## 预处理与标签

Aachen 原始 GNSS/星历/RTCM 用于钟差、轨道和大气相关修正；GNSS/IMU/LiDAR 状态估计生成用于标注的参考残差与地图。作者用 LiDAR 天空遮挡掩膜和预定义伪距残差阈值联合判断 NLOS。用于标签的多传感器参考残差，与输入特征中仅由 GNSS 最小二乘计算的残差必须分开。

论文 HK 数据来自其引用 [26] 的 2021 研究；HK 长度 10,573 s、最多 25 星，AC 长度 2,366 s、最多 8 星。它们不是本项目四段 2023 KLT 数值表，不能因都在香港就当作相同数据。

| 原论文输入/目标 | 当前 KLT 特征表 | 复现需要 |
| --- | --- | --- |
| Elevation | elevation_deg 有值 | 核对单位和计算坐标 |
| Azimuth | 缺失 | 用同源卫星星历、接收机位置重算；不能填常数冒充 |
| C/N0 | cn0_dbhz 有值，零占位转 NaN | 区分缺测；归一化只拟合训练集 |
| LS 伪距残差（m） | residual_normalized 为归一化残差 | 获取原始残差及定义；不能直接反推米值 |
| 残差 RSS | 缺失；cn0_std_dbhz 是信号标准差 | 由原始 LS 残差按声明的历史窗计算；不是 std 的替代名 |
| 全部卫星的共同时间窗口 | 时间/卫星 ID 已有 | 构造 (N,T,F)、缺失掩膜和稳定卫星映射；遮挡/新升星不得变成旧星历史 |
| 独立 LOS/NLOS 标签 | 鱼眼相机标签已有 | 与作者 LiDAR+残差标签有定义差异，需披露 |
| 伪距误差回归真值 | 缺失 | 同源参考轨迹、卫星坐标、钟差/改正项；不能用输入 LS 残差作独立误差真值 |

当前额外的 Doppler/伪距变化率一致性与 C/N0 波动是有意义的工程特征，但不是原论文五特征。正文把 Doppler rate consistency 列为未来扩展。

## 数值核对与解释

以下是原论文 NLOS 类结果，不是本项目成绩：

| 训练/测试情形 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 作者模型 HK 同域 | 0.80 | 0.79 | 0.80 |
| 作者模型 AC 同域 | 0.51 | 0.42 | 0.46 |
| 作者模型 HK 训练、AC 测试 | 0.40 | 0.60 | 0.48 |
| XGBoost HK 同域 | 0.77 | 0.80 | 0.78 |

复杂模型在另一数据集也显著退化；不能用 HK 的 0.80 保证 KLT 或其他设备同样表现。AC 的 NLOS 比例只有 18.87%，作者模型总体 Accuracy 0.81 也不能掩盖 NLOS Recall 0.42。文章没有给出多随机种子的区间或精确同域划分细节，因此分数差距不等于显著优越的统计证明。

Düsseldorf 无 NLOS 真值；Table VII 是预测比例，不是正确率。其排除观测后的轨迹改善提供定位一致性证据，正文仍说明位置精度不能保证。不能从预测排除比例推算漏报率，也不能把轨迹视觉效果当作完整定位精度指标。

## 当前可执行边界与下一步

本项目已经运行的单星因果 Transformer 测试 F1=0.712，见[本地实验](../experiments/transformer_nlos.md)。它没有跨卫星注意力、BiLSTM 和独立伪距误差头，因此不属于这篇论文的复现。

最合适的实施顺序是先从同源 KLT 原始 RINEX、星历、参考轨迹与标签生成论文所需特征，再建立全部卫星的历史张量。随后按论文结构实现一个完整模型，保存未明确超参数的独立选择；LR/RF 与 SVM/XGBoost 可作为同协议基线。用户当前不要求组件消融，也不要求 MMTNet。

若先在已有五特征上开发跨卫星 LSTM/注意力，只能命名为“Zhang2023-inspired 分类适配版”，并明确无伪距误差监督；不能与论文原结果作复现差值比较。论文原样结果复现需要作者代码/超参数、HK/AC 原数据和确切划分。当前 Reproducibility verdict 为 **CANNOT_VERIFY**，不是断言论文不能复现。

选模继续使用验证 Precision/Recall/F1，测试只报告冻结方法。当前 KLT 测试已查看，用于后续方法开发时应称开发阶段留出评估；进一步调优和最终成绩需要新增未查看数据。多星误报排除会削弱几何与观测冗余，未来同源定位验证也要检查可用卫星数、DOP、失解比例和 Median/P95，不能只追求高 NLOS Recall。

## 方法解释检查（11/11）

| 检查 | 此次判断 |
| --- | --- |
| Simpson 聚合反转 | 未提供充分逐段数据，无法检验；HK/AC 不混成一个分数 |
| 生态谬误 | 区域总体成绩不外推每颗卫星/每种遮挡 |
| 选择偏差 | 仅可跟踪且有标签观测，不能外推完全失锁 |
| Collider 控制偏差 | 未进行因果控制分析；没有建立因果模型 |
| 基准率忽略 | 明列 AC NLOS 18.87%，区分 Accuracy 与 Recall |
| 回归均值 | 没有把重新评估低分样本当成算法提升证据 |
| 幸存者偏差 | 跟踪与标注筛选可能影响覆盖，全文不足以量化 |
| 多重搜索 | 未提供全部调参轨迹；不能声称差异显著 |
| 分析路径选择 | 训练协议缺项，独立实现需逐项声明，不能伪称精确复现 |
| 相关当因果 | 注意力权重/特征重要性不直接证明遮挡原因；定位收益需同源实验 |
| 反向因果 | RSS/残差是观测结果指标，不能由模型得分确认物体类型 |

这是来源和可复现条件分析，无成功重跑作者命令，因此状态为 ANALYZED。没有新增论文模型训练结果、置信区间或显著性检验。
