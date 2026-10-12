# 模型卡

版本 0.3。NavDiag 用于离线信号分类、定位分析与有依据的问答。当前实测数据为香港 KLT 和 UrbanNav；未验证跨城市部署或导航安全性能。

## 卫星 LOS/NLOS 识别

一条样本为采集段、设备、时间、卫星的组合。正类 NLOS=1，负类 LOS=0；unknown 不参与监督和评分，可用观测仍可保留在因果历史中。参考位置、相机/建筑图、标签和区域 ID 不作为学习输入。

五项特征为 `cn0_dbhz`、`elevation_deg`、`residual_normalized`、`pr_rate_consistency_mps`、`cn0_std_dbhz`。非正 C/N0 转为缺测；填充与标准化仅拟合训练集。

| 模型 | 输入与固定配置 |
| --- | --- |
| C/N0 规则 | 当前 C/N0 <30 dB-Hz，无训练参数 |
| LR | 当前五特征，balanced，C=1 |
| RF | 当前五特征，200 树、深度12、最小叶8、balanced |
| MLP、CNN、LSTM、GRU、Transformer | 同星过去及当前32次观测；宽64、两层、dropout 0.1 |

五网络统一 AdamW，学习率0.0005、weight decay0.01、batch128，最多30轮，验证 BCE 早停 patience6；CNN 核5，Transformer 四头。窗口跨采集段/设备/卫星或 >5 秒间隔重置，无未来观测。树模型与神经网络的历史输入不同，因此是完整分类器对比。

分类阈值最大化验证 F1；验证损失选择神经网络检查点；平均验证 F1 选择汇总模型族。测试分数只用于报告。KLT 六折使用配对验证/测试段，UrbanNav 六方向使用完整城区互斥，另有冻结源模型外部迁移。

KLT 九段共2851历元、45228卫星行；UrbanNav 77744行中27839行通过几何参考保护。香港相邻区域、单次采集、日期混杂与 unknown 筛选限制结论，部分测试记录在开发中已查看。详见[标签](OBSTRUCTION_LABELS.md)、[KLT](experiments/nlos_protocols.md)、[UrbanNav](experiments/urbannav_generalization.md)。

## 定位失准告警

一条样本为接收机定位历元，正类为基线 SPP 水平误差 >10 m。10 m 是任务定义，精度另用 Median/P95/RMSE 与误差内比例评估；告警不表示已识别 NLOS 或滤波后失准。

输入为卫星数、C/N0 均值/最小值、HDOP、残差 RMS 和因果估计速度。比较固定规则、balanced LR、150树/深度8/最小叶6的 RF，以及 hidden24、10次历史的 GRU。GRU 使用 Adam0.002、batch128、最多20轮、patience5；历史间隔 >5 秒重置。训练覆盖不足50%的特征剔除，其余训练侧填充和标准化。

三条完整路线轮流留出，阈值与模型只由各轮验证段选择。参考位置只用于标签与评分；全部成功 SPP 点进入滤波/GRU 历史，无参考仍保留输入。主实验为尖沙咀训练、黄埔验证、旺角测试。[定位报告](experiments/positioning_expansion.md)同时报告误差、误报/漏报和可用率。

## 定位与滤波

UrbanNav 使用 pyrtklib0.2.7 调用广播 GPS/北斗单频 SPP，10°仰角，BRDC 电离层/SAAS 对流层改正。ENU 原点来自首个成功 SPP。固定与自适应平面 KF 的 sigma/过程噪声从八组候选按验证 RMSE 选择；风险五箱方差映射是经验权重，未标定为失准概率。

Doppler KF 加入 RTKLIB 速度与正定协方差，旋转至 ENU；位置/速度 NIS 门限100/25，速度协方差下限0.04 (m/s)²，间隔 >10 秒重置。位置/速度相关性近似为零，不等价于 FGO、TDCP 或完整 GNSS/IMU。仅原有成功 SPP 点输出，未验证跨断解定位。参见[方法来源](research/positioning_methods.md)。

## 使用边界

当前位置分支尚无“逐卫星 NLOS 预测 → 观测加权 → 定位收益”的实测闭环；Zhang 2023 完整联合网络未复现。旺角 SPP 可用率17.2%，RMSE15.481 m，当前滤波不能保证持续改善。静态建筑标签不能区分树木、车辆等遮挡物，LOS 也可能有多径。

历史 GSDC 适配器保留作解析示例：修正时间与高度后4572匹配历元，短留出仅11点且无正类，旧验证 F1=0.991 已撤回。其结果不计入当前主实验。密钥、数据与可信本地模型产物不进入发布包；joblib 仅加载可信文件。
