# UrbanNav 跨城区数据适配器

本目录现也包含定位/告警工具：`prepare_positioning.py`生成完整原始可用率表、全部成功观测历史与独立参考评分表；`prepare_velocity.py`从RTKLIB提取Doppler速度及协方差，严格核对已有SPP人口不变。它们仍为独立GPL工具，不随MIT wheel发布。固定实验、结果和方法出处见[完整路线评估](../../docs/experiments/positioning_expansion.md)。

此目录是独立的 GPL-3.0-or-later 工具，不随主 Python 包发布。它调用固定 SHA-256 的外部 [Fossett GNSS 特征提取器](https://github.com/fossettlab/gnss-nlos-notebook)，保持与 KLT 实验相同的五项 GPS/北斗无线电/SPP 特征定义。

```powershell
python -m pip install -e ".[gnss,temporal]"
python scripts/fetch_urbannav.py
python tools/urbannav/fetch_extractor.py
python tools/urbannav/prepare.py
python -m navdiag evaluate-nlos-regions
```

最后一步还需要之前 `compare-nlos-protocols` 生成的 KLT 源模型，路径固定在 `config/nlos_generalization.json`。只运行固定参数训练和冻结阈值的外部测试，不进行超参数搜索。

数据来自 [UrbanNav 作者仓库](https://github.com/IPNL-POLYU/UrbanNavDataset)、[NOAA 广播星历](https://ngs.noaa.gov/CORS/data.shtml) 与 [香港 SatRef](https://www.geodetic.gov.hk/en/rinex/downv.aspx)。UrbanNav README 没有给出明确的数据再分发许可证；SatRef 页面限制为个人、非商业内部使用。原始数据、派生逐卫星表和模型文件均留在 Git 忽略目录中，公开仓库只包含适配代码、下载清单和汇总报告。

标签来自独立位置真值、广播星历及静态建筑遮挡图，明确标记为 `static_3d_sky_mask`。它们是几何参考标签，不是人工逐卫星接收真值。固定规则排除 Q>2、位置不匹配、距网格>2m、高度差>3m，以及建筑边缘附近的样本。四个邻近网格、相邻方位角和公开文件的 0°/360°差异共同组成保守包络，仰角位于包络边缘±5°内标记为 unknown。这些间隔是工程保护范围，没有经人工真值校准，不能据此声称标签绝对准确。

`navdiag.visibility` 实现独立的几何参考计算。位置真值、遮挡图、方位角、区域与质量字段只供审计，均不在模型五项输入特征中。详情见 `docs/experiments/urbannav_generalization.md`。
