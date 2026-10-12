# 定位与告警的方法来源

核对日期：2026-10-12。区分直接调用、方法参考和待复现；引用论文不等于复现它的算法。

| 来源 | 阅读与采用内容 | 当前接入程度 |
| --- | --- | --- |
| [RTKLIB pntpos.c](https://github.com/tomojitakasu/RTKLIB/blob/rtklib_2.4.3/src/pntpos.c) / [pyrtklib](https://github.com/IPNL-POLYU/pyrtklib) | 伪距 SPP、广播星历/钟差/大气改正；`estvel` 从 Doppler 估计 ECEF 速度并输出 `qv` | 直接调用本机 pyrtklib 0.2.7；检查成功状态、卫星数量、10°仰角、有限值；速度和协方差旋转到 ENU。所浏览上游分支是方法核对来源，未冒充本机二进制的编译提交 |
| [Fossett GNSS notebook](https://github.com/fossettlab/gnss-nlos-notebook/blob/b13aa856f723a472bd35d45171017580642054cf/extract_features.py) | RINEX 读取、完整历元分组、rover 观测、GPS/北斗与时间适配 | 独立 GPL 工具调用固定 SHA 的上游辅助函数；不随 MIT wheel 分发 |
| [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib)，Wen/Hsu ICRA 2021 | 历史与当前伪距、Doppler 联合 FGO；README 提醒代码与论文不完全相同 | 参考速度约束和完整历史输入；新增因果平面 KF，**未运行 GraphGNSSLib 或完成 FGO 复现**。原项目依赖 ROS/Ceres |
| [Suzuki 2023](https://www.mdpi.com/1424-8220/23/3/1205)，相关开源 [gsdc2023](https://github.com/taroz/gsdc2023) | 区分速度/位置信息，结合 Doppler、TDCP 与两步优化；公开项目包含 GNSS/IMU FGO | 核对输入、指标与部分表格；当前落实 Doppler 对照，未实现 TDCP、完整 IMU 或两步优化。gsdc2023 是不同年份实现，不能声称复现 2023 论文；依赖 MATLAB/GTSAM/MatRTKLIB |
| [Wen/Zhang/Hsu TVT 2022](https://ira.lib.polyu.edu.hk/handle/10397/92728) | 观测级鲁棒 FGO-GNC，均值/最大误差与不同接收机 | 研究参照，尚未接入；2D MEAN 不能直接与本项目 RMSE 排名 |
| [Ozeki/Kubo 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.868608/full) | SVM 识别结合残差检查，识别到 DGNSS 的连接 | 指导后续同源 NLOS → 加权/受几何约束剔除；当前告警预测位置误差，尚不是逐卫星定位闭环 |

基础平面滤波、LR/RF/GRU 告警及五箱风险方差映射是本项目的可解释基线，并非上述论文原模型。告警仅输入卫星数、C/N0、HDOP、SPP 残差与观测估计速度；参考误差只构造 `>10 m` 标签。阈值与选模在验证采集段完成，测试不调参。

## 本轮落实

1. 三条完整 UrbanNav 路线分别测试一次，固定循环轮换训练/验证；记录原始历元、解算可用率、参考覆盖与断解，避免历元随机划分。
2. 滤波与 GRU 历史保留全部成功观测；参考匹配/质量仅选择监督和评分点，不能让真值可用性改变历史。
3. 新增位置 + Doppler 速度因果 KF。`qv` 正定检查并转 ENU；速度 NIS 门限 25、位置门限 100、10 秒断点重置、速度协方差特征值下限 0.04 `(m/s)²`。这些冻结工程值未经独立场景校准。
4. Doppler 只在原有成功 SPP 点输出，不补报断解定位；八个既有 sigma/过程噪声候选仅按验证 RMSE 选择。没有因新增 Doppler 再训练告警或 NLOS 模型。

位置/速度相关性近似为零，RTKLIB `qv` 也不是已验证的经验误差校准。这是有方法依据的新增基线，不能称为完整紧耦合 GNSS、INS 或 FGO。改善和退步均保留，见[完整路线实验](../experiments/positioning_expansion.md)。

后续优先做同源观测级闭环，再考虑 Doppler/TDCP 因子图、周跳与弧段管理。完整 GNSS/IMU 还需时间同步、姿态/重力/偏置/杆臂，不能接入加速度字段便称完成。

作者 gsdc2023 README 另提供约 2.7 GB 预处理数据入口，可继续核查。本轮未下载，尚未确认其中训练真值、路线数和再分发条件，不计入当前测试规模。
