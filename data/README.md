# 数据来源、字段与处理

项目区分“接收机历元表”和“时间×卫星表”。下文先说明定位历元与历史GSDC适配，再说明当前KLT/UrbanNav识别；两者标签和输入不能互换。

每行是一个定位历元。`schema.csv` 只含字段名。模型输入是卫星数、平均/最小 C/N0（dB-Hz）、HDOP、WLS 残差 RMS（m）和因果估计速度（m/s）。`trip_id` / `device_id` 用于分组，`timestamp_ms` 是 UTC Unix 毫秒；`data_kind` 区分真实/模拟。`horizontal_error_m` 是相对独立参考轨迹的水平误差，只生成标签和离线指标。

## 三种运行方式

- `demo`：人为构造特征与标签的教学样例。
- `demo --navigation`：模拟卫星几何、码伪距和 ENU 加速度，实际运行定位算法；仍是 synthetic。
- `prepare-gsdc`：解析真实公开派生观测和参考轨迹，输出 real 历元表。

## 公开文件

来源为 [Stanford NavLab deep_gnss](https://github.com/Stanford-NavLab/deep_gnss) 与 [gnss_lib_py](https://github.com/Stanford-NavLab/gnss_lib_py)。具体版本与文件 SHA-256 在 `config/public_sources.json`；下载后副本与清单只保存于忽略的 `data/raw/`。

| 本地采集组             | 格式 / 设备                 | 匹配历元 | 当前用途 |
| ---------------------- | --------------------------- | -------: | -------- |
| 2021-04-22-US-SJC-1    | GSDC 2021 / SamsungS20Ultra |    2,824 | 训练     |
| 2020-05-14-US-MTV-1    | GSDC 2021 / Pixel4          |    1,737 | 验证     |
| published-sample       | GSDC 2022 parser example    |        6 | 流程测试 |
| 2023-09-07-18-59-us-ca | GSDC 2023 / pixel7pro       |        5 | 流程测试 |

2022 格式例子的时间属于 2021-04-29，原始完整行程名没有从示例元数据中确定；保留 `published-sample` 名称。这里的年份是格式/数据发布标签，不必然是采集年份。两个短示例不是完整驾驶实验。训练/验证行程名称在上游目录中的分配不构成本项目的最终测试方案；本项目明确使用 `config/public_split.json`。

## 处理顺序

1. 统一旧版/新版字段；GSDC 2021 派生观测映射到前一个实际记录历元，首历元剔除，并要求发射至接收时间差在 (0,300) ms。再按 18 秒闰秒差将 GPS 毫秒转为 UTC；新版 UTC 原样使用。参考时间不平移，2021 参考高度减 61 m，新版高度原样使用。依据与影响见[修正记录](../docs/experiments/gsdc2021_correction.md)。
2. 修正伪距：raw + satellite clock − inter-signal bias − ionosphere − troposphere。
3. 过滤缺失/不合理必需字段。同一历元每颗星保留不确定度最小的一条码观测，不做双频组合。
4. 估计位置和每个星座时钟；WLS 使用供给的不确定度，另算等权与 Huber 稳健基线。稳健失败时记录回退。
5. 只用观测提取特征和因果速度。2021 文件没有 C/N0，置 NaN，不补造信号值。
6. 用第一份 WLS 坐标定义 ENU 原点，参考 ECEF 插值只在支持区间和最大间隙内计算标签，不外推。
7. 按采集行程分组所有设备；训练集覆盖选择特征、拟合填充；验证集选择模型/阈值/滤波参数。

`preprocess_report.json` 保存源文件指纹、成功/失败历元、参考剔除、稳健回退和原点来源。CLI `train` 接收处理后的历元表，不能直接传原始卫星行表。

真实公开文件没有可用于本项目已标定 ENU 接口的 IMU 数据，因此本次真实实验不声称 GNSS/IMU 融合。未来接入完整数据时，要单独实现原始 IMU 姿态和标定流程。

## 新增卫星级 LOS/NLOS 数据

`prepare-klt` 消费 [Fossett Laboratory 的 KLT 数值特征表](https://github.com/fossettlab/gnss-nlos-notebook)，标签来自 [原始 KLT 数据集](https://github.com/ebhrz/KLTDataset)的鱼眼相机天空掩膜。每行是时间 × 卫星，与上面的定位历元表不同；字段见 `satellite_schema.csv`。

固定来源在 `config/nlos_sources.json`。下载无需账号，原始数据与 GPL-3.0 LICENSE 保存在 data/raw/klt_features，处理表在 data/processed/klt，均被 Git 忽略。当前表只有四个 2023-11-09 短采集段，共 26,522 行。config/klt_split.json 按 KLT2/KLT1/KLT3 分别训练/验证/测试，同一区域不跨集合。

`signal_label` 为 los/nlos/unknown，`signal_label_source` 保留来源；`scene_label` 仅为 urban_unspecified，`obstruction_type` 为 unknown。高楼/树木等具体标签目前没有真值。独立参考坐标、标签、分组 ID 不作为学习特征。详见 [标签协议](../docs/OBSTRUCTION_LABELS.md)和[开源复用记录](../docs/OPEN_SOURCE_REFERENCES.md)。

## 使用与再分发

双协议主实验扩展到九段、三个日期，2,851 个历元和 45,228 条卫星观测。`scripts/fetch_klt_extended.py` 根据固定来源及校验和下载五段新增标签、三个日期观测与星历；`tools/klt/prepare.py` 用统一流程重算全部九段特征并逐历元检查时间对齐。原作者声明不可靠的 `1116_KLT3_649` 排除。原始文件、派生表与模型产物继续留在 Git 忽略目录，公开代码只包含下载说明、来源指纹和实验计划。

三对完整采集段划分在 `config/nlos_protocols.json`；同域覆盖测试区域的其他采集段，跨区域排除目标区域的全部训练和验证数据。目标比例 6∶2∶2，实际按完整段略有偏差，详见[实验设计](../docs/experiments/nlos_protocols.md)。

第三方原始数据及精确轨迹不进入代码仓库。本项目 MIT 许可仅覆盖原创代码；上游代码库的 MIT 许可也不自动替代 GSDC 数据发布/竞赛条款。扩展使用应核对 [GSDC 发布页](https://www.kaggle.com/competitions/smartphone-decimeter-2023) 和原发布方条款。

解析方式参考 [gnss_lib_py 原始教程](https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_google_decimeter_notebook.html)。本项目实现独立定位基线，并未声称复现上游完整算法或排行榜成绩。

## UrbanNav 新城区记录

公开三城区记录已按 [固定源清单](../config/urbannav_sources.json) 下载到忽略的 `data/raw/urbannav`，独立位置/建筑图几何标签与同源五项无线电特征生成于 `data/processed/urbannav`。原始 5,671 历元、77,744 卫星行，27,839 行通过参考质量、时间、网格、高度和边缘检查；其余 unknown 不进入分类评分。标签方法、覆盖、固定模型实验及许可边界见 [UrbanNav 报告](../docs/experiments/urbannav_generalization.md)。所有区域都在香港，未验证跨城市。

## UrbanNav 定位与告警表

独立定位分支使用同一批原始GNSS/参考采集，生成在 `data/processed/urbannav_positioning_v3`：`observations.csv`保留全部成功SPP观测，`epochs.csv`保留合格参考监督/评分行，`availability.csv`记录所有原始历元的解算与参考状态。原点来自首个成功SPP，滤波/GRU历史保留无合格参考的成功观测；真值仅用于监督、验证和评分。

三段5671原始历元中，1193点同时具有成功解和合格参考。旺角3350原始点、577有效解、1270合格参考、139可评分点，分母分别报告。`data/processed/urbannav_doppler/velocity.csv`另存RTKLIB速度和协方差，不改变告警六项输入或原评分人口。

当前整段划分、三路线轮换、Doppler提取和复查见[完整定位报告](../docs/experiments/positioning_expansion.md)与[方法来源](../docs/research/positioning_methods.md)；GSDC的11点历史样例不与此分支池化。
