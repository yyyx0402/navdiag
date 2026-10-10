# 数据来源、字段与处理

每行是一个定位历元。`schema.csv` 只含字段名。模型输入是卫星数、平均/最小 C/N0（dB-Hz）、HDOP、WLS 残差 RMS（m）和因果估计速度（m/s）。`trip_id` / `device_id` 用于分组，`timestamp_ms` 是 UTC Unix 毫秒；`data_kind` 区分真实/模拟。`horizontal_error_m` 是相对独立参考轨迹的水平误差，只生成标签和离线指标。

## 三种运行方式

- `demo`：人为构造特征与标签的教学样例。
- `demo --navigation`：模拟卫星几何、码伪距和 ENU 加速度，实际运行定位算法；仍是 synthetic。
- `prepare-gsdc`：解析真实公开派生观测和参考轨迹，输出 real 历元表。

## 公开文件

来源为 [Stanford NavLab deep_gnss](https://github.com/Stanford-NavLab/deep_gnss) 与 [gnss_lib_py](https://github.com/Stanford-NavLab/gnss_lib_py)。具体版本与文件 SHA-256 在 `config/public_sources.json`；下载后副本与清单只保存于忽略的 `data/raw/`。

| 本地采集组 | 格式 / 设备 | 匹配历元 | 当前用途 |
| --- | --- | ---: | --- |
| 2021-04-22-US-SJC-1 | GSDC 2021 / SamsungS20Ultra | 2,825 | 训练 |
| 2020-05-14-US-MTV-1 | GSDC 2021 / Pixel4 | 1,739 | 验证 |
| published-sample | GSDC 2022 parser example | 6 | 流程测试 |
| 2023-09-07-18-59-us-ca | GSDC 2023 / pixel7pro | 5 | 流程测试 |

2022 格式例子的时间属于 2021-04-29，原始完整行程名没有从示例元数据中确定；保留 `published-sample` 名称。这里的年份是格式/数据发布标签，不必然是采集年份。两个短示例不是完整驾驶实验。训练/验证行程名称在上游目录中的分配不构成本项目的最终测试方案；本项目明确使用 `config/public_split.json`。

## 处理顺序

1. 统一旧版/新版字段；2020–2024 数据的 GPS 毫秒按 18 秒闰秒差转换 UTC，新版有 UTC 字段时直接使用。
2. 修正伪距：raw + satellite clock − inter-signal bias − ionosphere − troposphere。
3. 过滤缺失/不合理必需字段。同一历元每颗星保留不确定度最小的一条码观测，不做双频组合。
4. 估计位置和每个星座时钟；WLS 使用供给的不确定度，另算等权与 Huber 稳健基线。稳健失败时记录回退。
5. 只用观测提取特征和因果速度。2021 文件没有 C/N0，置 NaN，不补造信号值。
6. 用第一份 WLS 坐标定义 ENU 原点，参考 ECEF 插值只在支持区间和最大间隙内计算标签，不外推。
7. 按采集行程分组所有设备；训练集覆盖选择特征、拟合填充；验证集选择模型/阈值/滤波参数。

`preprocess_report.json` 保存源文件指纹、成功/失败历元、参考剔除、稳健回退和原点来源。CLI `train` 接收处理后的历元表，不能直接传原始卫星行表。

真实公开文件没有可用于本项目已标定 ENU 接口的 IMU 数据，因此本次真实实验不声称 GNSS/IMU 融合。未来接入完整数据时，要单独实现原始 IMU 姿态和标定流程。

## 使用与再分发

第三方原始数据及精确轨迹不进入代码仓库。本项目 MIT 许可仅覆盖原创代码；上游代码库的 MIT 许可也不自动替代 GSDC 数据发布/竞赛条款。扩展使用应核对 [GSDC 发布页](https://www.kaggle.com/competitions/smartphone-decimeter-2023) 和原发布方条款。

解析方式参考 [gnss_lib_py 原始教程](https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_google_decimeter_notebook.html)。本项目实现独立定位基线，并未声称复现上游完整算法或排行榜成绩。
