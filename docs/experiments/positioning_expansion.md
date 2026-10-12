# 完整路线定位与告警评估（2026-10-12）

新增 UrbanNav 尖沙咀、黄埔、旺角三段原始 GNSS 与独立 SPAN 参考。主实验为尖沙咀训练、黄埔验证、旺角测试；补充三轮固定角色循环，每条路线测试一次。固定规则、LR、RF、GRU 使用原默认参数；滤波八组既有噪声候选、分类阈值和 GRU 早停只使用各轮验证段。NLOS 调参保持停止。

数据以前已用于 NLOS 研究，主路线结果也在补充轮换前查看过，因此是开发阶段整段留出，**不是全新盲测或跨城市验证**。区域与日期共同变化；1193 个相关点不是 1193 个独立实验。

## 数据分母

| 路线 | 采集时间 | 原始历元 | 有效单点解 | 解可用率 | 时间匹配参考 | 质量合格参考 | 可评分点 | 最长断解 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 尖沙咀 | 13.1 min | 786 | 351 | 44.7% | 786 | 646 | 311 | 103 s |
| 黄埔 | 25.6 min | 1535 | 887 | 57.8% | 1535 | 1317 | 743 | 87 s |
| 旺角 | 55.8 min | 3350 | 577 | 17.2% | 2312 | 1270 | 139 | 464 s |

共 5671 个原始历元，约 94.5 分钟，1193 个可评分点。旺角公开真值只覆盖部分时间；Q=1/2 与 ±0.05 秒匹配进一步限制分母。其质量合格参考中可用率为 139/1270=10.9%，全段可评分比例为 139/3350=4.15%；两者都不同于全段 SPP 可用率 17.2%。

断解率描述**当前接收机、星历、GPS/北斗单频、10°仰角与 RTKLIB 策略**，不是 NLOS 比例或数据集固有可用率。失败包括求解/有效性检查未通过，尚未隔离物理原因。低可用率已阻止当前方案被称为连续可靠定位。

全部成功 SPP 点进入滤波与 GRU 历史，无合格参考仍保留输入；真值只用于监督、验证和评分。原点来自首个成功 SPP。滤波间隔超过 10 秒、GRU 超过 5 秒重置；旺角形成 43/63 段。未给失败 SPP 历元补报有效位置或告警。

## 定位结果

同轮同一评分人口的水平 RMSE（米）：

| 测试路线 | RTKLIB SPP | 固定 KF | 风险自适应 KF | Doppler 速度约束 KF |
| --- | ---: | ---: | ---: | ---: |
| 尖沙咀，311 点 | 7.881 | 9.559 | 6.902 | 8.531 |
| 黄埔，743 点 | 12.153 | 15.848 | 12.285 | 11.170 |
| 旺角，139 点 | 15.481 | 15.669 | 15.751 | 14.610 |

自适应不普遍更优；Doppler 降低黄埔/旺角 RMSE，尖沙咀变差。旺角 Doppler Median/P95 为 8.979/33.151 m，SPP 为 7.563/28.046 m：**RMSE 改善不代表中位或尾部误差同步改善**。不根据测试重调阈值或删困难点。

旺角自适应的 ≤2 m 为 11.5%（139 个可评分点中的16个）；以全部1270个合格参考历元为分母，“有输出且≤2 m”仅1.26%。Doppler 仍用相同人口，不提升解算可用率。不能用条件误差掩盖断解。

Doppler KF 参考 RTKLIB、GraphGNSSLib 与文献的速度思路，是轻量对照，非 TDCP/FGO/完整 GNSS-IMU 复现；出处与相关性近似见[方法来源](../research/positioning_methods.md)。

## 告警结果

任务是 **SPP 水平误差 >10 m**，不是 LOS/NLOS，也不是滤波后误差告警。三条测试路线等权平均：

| 模型 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 固定规则 | 0.658 | 0.069 | 0.108 |
| LR | 0.507 | 0.807 | 0.609 |
| RF | 0.485 | 0.805 | 0.583 |
| GRU | 0.497 | 0.778 | 0.597 |

各验证段分别选出旺角 RF、尖沙咀 GRU、黄埔 LR；宏平均未用于重选主模型。RF 非所有告警场景最佳，不能套用 NLOS 的旧排名。

主实验旺角55个正样本：RF阈值0.25来自黄埔验证，正确告警45、漏报10、误报40、正确排除44，Precision0.529、Recall0.818、F1 0.643。误报仍多；结果仅针对139个可评分点，断解时并未验证有效告警。

## 复查与复现

汇总、SHA、冻结设置与诊断见[公开汇总 JSON](positioning_expansion_report.json)。原始数据、参考轨迹、逐历元表、模型权重留在 Git 忽略目录。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gnss,temporal]"
.\.venv\Scripts\python.exe scripts/fetch_urbannav.py
.\.venv\Scripts\python.exe tools/urbannav/fetch_extractor.py
.\.venv\Scripts\python.exe tools/urbannav/prepare_positioning.py
.\.venv\Scripts\python.exe scripts/run_positioning_expansion.py
.\.venv\Scripts\python.exe scripts/run_positioning_rotation.py
.\.venv\Scripts\python.exe tools/urbannav/prepare_velocity.py
.\.venv\Scripts\python.exe scripts/run_doppler_comparison.py
.\.venv\Scripts\python.exe scripts/review_positioning.py
Copy-Item data/processed/urbannav_positioning_v3/availability.csv artifacts/urbannav_positioning/availability.csv
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/urbannav_positioning --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization
```

遵循来源条款；下载器也获取原主线需要的遮挡图。冻结记录对应当时源文件/实现与环境，后续变更应使用新输出目录，不能改写指纹伪装原运行。已完成目录拒绝覆盖。旧 GSDC 11 点结果保留在[历史审计](../research/positioning_audit.md)，未与 UrbanNav 不同硬件/算法池化。
