# 定位方法来源

| 来源 | 当前使用方式 |
| --- | --- |
| [RTKLIB pntpos](https://github.com/tomojitakasu/RTKLIB/blob/rtklib_2.4.3/src/pntpos.c) / [pyrtklib](https://github.com/IPNL-POLYU/pyrtklib) | 直接调用 pyrtklib0.2.7 的广播伪距 SPP、Doppler 速度与协方差；浏览分支不代表本机二进制编译提交 |
| [Fossett 提取器](https://github.com/fossettlab/gnss-nlos-notebook/blob/b13aa856f723a472bd35d45171017580642054cf/extract_features.py) | GPL 独立工具调用固定提交的 RINEX/时间/观测辅助函数 |
| [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib) | 参考历史伪距与 Doppler 联合估计思想，未运行其 FGO；上游依赖 ROS/Ceres |
| [Suzuki 2023](https://www.mdpi.com/1424-8220/23/3/1205) / [gsdc2023](https://github.com/taroz/gsdc2023) | 参考速度约束；未实现 TDCP、完整 IMU 或两步优化，开源实现年份不同，不构成论文复现 |
| [Wen/Zhang/Hsu 2022](https://ira.lib.polyu.edu.hk/handle/10397/92728) | 观测级鲁棒 FGO-GNC 研究参照，未接入；论文 MEAN 与本项目 RMSE 不可直接排名 |
| [Ozeki/Kubo 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.868608/full) | NLOS 识别结合残差检查的定位闭环参照；当前未完成闭环 |

## 当前实现

固定/风险自适应平面 KF、LR/RF/GRU 告警与五箱风险方差是项目基线。告警预测基线水平误差 >10 m；参考误差只用于监督、验证和评分。

Doppler KF 将 RTKLIB ECEF 速度与正定 qv 转至 ENU，速度协方差特征值下限0.04 (m/s)²，位置/速度 NIS 门限100/25，10秒断点重置。八组 sigma/过程噪声候选只按验证 RMSE 选择；这些工程保护值未经独立场景校准。

位置/速度相关性近似为零，qv 未做经验误差校准。全部成功 SPP 进入因果历史，只有原有成功解输出，未计跨断解收益。此实现是轻量速度约束对照，不是 TDCP、FGO 或完整 GNSS/INS。

当前结果与可用率见[完整路线评估](../experiments/positioning_expansion.md)。未来同源 NLOS 加权还需检查观测冗余、几何与失解；完整 GNSS/IMU 需同步、姿态、重力、偏置和杆臂处理。
