# GSDC 2021 时间与参考高度修正（2026-10-11）

旧实现遗漏 2021 派生观测时间和参考高度修正，造成参考误差、失准标签和相关评估错误。旧真实 GRU 验证 F1 0.991 已撤回，不能用于简历或算法效果声明。历史数值保留在标为 invalid 的 [旧 JSON](v0_2_public_report.json)；当前快照见 [修正 JSON](gsdc2021_corrected_report.json)。模拟数据与独立 KLT LOS/NLOS 标签不受影响。

## 依据与实现

依据 [Stanford gnss_lib_py 固定版本解析器](https://github.com/Stanford-NavLab/gnss_lib_py/blob/3a5c43cdf9121b68414ec87cd3666b761f0faf8d/gnss_lib_py/parsers/google_decimeter.py)的 2021 主办方修正规则：

1. 将每个派生观测 epoch 映射到前一个实际记录 epoch，首 epoch 无前驱而剔除。记录间隔不一定等于 1,000 ms，不能统一减一秒。
2. 只保留映射后接收时间减卫星发射时间位于 (0,300) ms 的观测，再做 GPS/UTC 转换。
3. 2021 参考 heightAboveWgs84EllipsoidM 减 61 m；参考时间不平移。2022/2023 的 UTC 与高度不改变。

这些规则来自上游格式处理，没有按参考误差寻找最优时间偏移。诊断中尝试时间偏移只是发现异常，未作为修正方法。源码 `src/navdiag/gsdc.py`，测试包含不规则 epoch 间隔、发射时间异常与新版高度保持原样。

## 重算结果

使用相同原始文件、整行程划分、seed=42、默认 >10 m 标签和 20 轮上限的 GRU 重新处理、训练和选择模型。原始文件未修改。处理 4,572 个匹配历元：训练 2,824，验证 1,737，短解析测试 11。

| 行程 / 用途 | WLS Median (m) | P95 (m) | ≤2 m | ≤5 m | >10 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Samsung SJC / 训练 | 7.593 | 29.169 | 3.36% | 26.13% | 33.68% |
| Pixel4 MTV / 验证 | 1.351 | 3.062 | 76.34% | 98.96% | 0.29% |

Pixel4 旧 WLS 中位数约 25.455 m，修正后 1.351 m。这个变化主要是纠正数据对齐与评估，不是新模型提升定位精度；两次匹配数略有变化，不能视为严格同样本算法对比。

验证 >10 m 正样本仅 5 个。重算验证 F1：规则 0.045、LR 0.258、RF 0.103、GRU 0.308，选中 GRU。测试仍没有 >10 m 正样本，无法评估 Recall/F1；规则误报 6 次，LR/RF/GRU 为 0，但仅 11 个历元，不证明可靠检测。旧滤波参数由错误参考误差选择，其旧真实结果一并撤回；当前滤波在修正参考后重新验证选参。

## 为什么 10 m 不代表定位水平

`error > 10 m` 是粗粒度失准告警标签的默认定义，没有针对具体导航用途验证过；不能当作已达到的精度或可接受定位误差。精度应查看 Median/P95/RMSE，以及 1/2/5/10 m 内比例。网页定位表已增加 ≤2 m 和 ≤5 m 比例。

把标签改为 >2 m 不会改善坐标；本训练行程 >2 m 比例已达 96.64%，全报失准就可能得到很高 F1。任务阈值要按用途确定，同时核查类别比例和漏报/误报。现有真实定位样例仍太少；需要更多独立完整行程和新最终留出集。

```powershell
.\.venv\Scripts\python.exe -m navdiag prepare-gsdc --root data/raw --out data/processed/public_gsdc
.\.venv\Scripts\python.exe -m navdiag train --csv data/processed/public_gsdc/epochs.csv --out artifacts/public_gsdc --allow-missing-features --split-manifest config/public_split.json --with-gru --gru-epochs 20
```
