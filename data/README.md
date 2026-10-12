# 数据与字段

原始数据与派生表均在Git忽略目录，下载器使用固定URL、字节数和SHA-256。来源条款见[复用记录](../docs/OPEN_SOURCE_REFERENCES.md)。

| 数据 | 规模 / 当前用途 | 入口 |
| --- | --- | --- |
| KLT | 九段、2851历元、45228卫星行；同域/跨区域分类 | [工具](../tools/klt/README.md) |
| UrbanNav卫星表 | 三城区、77744行；27839有参考标签，其余unknown | [工具](../tools/urbannav/README.md) |
| UrbanNav定位表 | 5671原始历元、1193可评分点；整段留出与滤波 | [评估](../docs/experiments/positioning_expansion.md) |
| GSDC公开解析样例 | 修正后4572匹配点；短留出11点，不作主实验 | `scripts/fetch_public_data.py`、`prepare-gsdc` |

## 卫星表

每行是采集段×设备×时间×卫星；字段见[satellite_schema.csv](satellite_schema.csv)。正类NLOS，负类LOS，unknown不参与监督/评分。五特征为C/N0、仰角、归一化残差、伪距变化率一致性、历史C/N0标准差。标签、参考位置、区域ID和具体遮挡物不作学习输入。来源与评分保护见[标签说明](../docs/OBSTRUCTION_LABELS.md)。

## 定位表

每行是接收机历元；[schema.csv](schema.csv)仅含字段名。告警输入为卫星数、C/N0均值/最小值、HDOP、残差RMS、因果速度；`horizontal_error_m`仅生成 >10m 标签及评分。时间统一UTC Unix毫秒，参考不作学习输入。

UrbanNav `data/processed/urbannav_positioning_v3` 内，`observations.csv`保留全部成功SPP历史，`epochs.csv`保留参考合格监督/评分行，`availability.csv`保留所有原始历元状态。原点来自首个成功SPP。Doppler速度/协方差单独存于 `urbannav_doppler/velocity.csv`，不改变告警输入或评分人口。

GSDC适配统一旧/新版字段：2021观测映射前一实际记录历元，要求传播时间在(0,300)ms，GPS→UTC减18秒、参考高度减61m；新版UTC/高度原样使用。修正伪距为 raw+satellite clock−inter-signal bias−ionosphere−troposphere；同星保留最小不确定度码观测，运行WLS/等权/Huber对照。缺C/N0保留NaN。参考插值不外推，填充/标准化仅训练侧拟合。

`demo`和`demo --navigation`始终标为synthetic。模型权重、原始轨迹和第三方数据不打包发布；代码许可不代替数据条款。
