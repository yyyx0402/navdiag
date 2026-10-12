# 本地产物

子目录由Git忽略；新克隆仓库需自行生成数据和模型。

| 目录 | 内容 |
| --- | --- |
| `klt_protocols` | KLT六折报告、模型与预测 |
| `urbannav_generalization` | 冻结迁移与城区互斥报告 |
| `urbannav_positioning` | 定位/告警主报告、轮换、Doppler与覆盖 |
| `demo` | 独立模拟示例 |

报告保存划分、数据指纹、阈值、指标与冻结配置。数据/实现变化时使用新目录，不覆盖已完成的冻结实验。`review_project.py`与`review_positioning.py`只检查或回放，不训练。公开汇总位于[结果文档](../docs/README.md)。仅加载可信本地joblib产物。
