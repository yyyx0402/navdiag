# Zhang 2023 方法与实现对照

[Zhang、Wang、Vallery，2023](https://arxiv.org/abs/2309.00480v2)，*Learning-based NLOS Detection and Uncertainty Prediction of GNSS Observations with Transformer-Enhanced LSTM Network*，DOI 10.1109/ITSC57777.2023.10422672。

论文逐卫星 LSTM 编码历史，经跨卫星多头注意力、BiLSTM 与 MLP 融合，联合预测可见性和伪距误差。分类采用 MSE、误差回归采用 L1。当前 NavDiag Transformer 仅做单卫星32次因果历史分类，没有跨卫星上下文或误差回归头，**未完成该论文复现**。

## 特征与标签

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

输入 GNSS 最小二乘残差与独立参考标注残差必须分开；不能用输入残差当作独立误差真值。论文 HK/AC 数据与当前 KLT/UrbanNav 不是同一实验人口。

## 论文数值

| 训练/测试情形 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 作者模型 HK 同域 | 0.80 | 0.79 | 0.80 |
| 作者模型 AC 同域 | 0.51 | 0.42 | 0.46 |
| 作者模型 HK 训练、AC 测试 | 0.40 | 0.60 | 0.48 |
| XGBoost HK 同域 | 0.77 | 0.80 | 0.78 |

以上为作者报告的 NLOS 成绩，不是 NavDiag 测量。AC NLOS 占比18.87%，总体 Accuracy0.81仍可能伴随 Recall0.42。Düsseldorf 无 NLOS 真值，其预测比例不是正确率。

## 可复现范围

正文未完整给出窗口长度、宽度/头数、调度、缺失卫星处理、标签阈值、精确划分和种子；完整张量轴与初始化仍需作者实现核对。BiLSTM 是否读取未来取决于实际窗口时间索引。

补齐同源输入、全部卫星掩膜和独立误差目标后可以实现结构适配，但需声明自行选择的参数。现有五特征上的适配不能称为原样复现，也不能用作者分数直接判断实现误差。概念说明见[NLOS 背景](../knowledge/nlos.md)。
