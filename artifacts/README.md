# 本地实验产物

子目录由Git忽略，新的仓库没有预生成数据或模型。已有本机读取当前目录；快速学习入口运行 `python -m navdiag demo --navigation --out artifacts/learning_demo`，详细安装见[README](../README.md)。

| 目录                              | 用途                                                       |
| --------------------------------- | ---------------------------------------------------------- |
| `klt_protocols`                   | 当前KLT双协议套件，`suite_report.json`与六折报告/预测      |
| `urbannav_generalization`         | 冻结迁移与六方向城区互斥识别，`generalization_report.json` |
| `urbannav_positioning`            | 当前真实定位/告警主报告、三轮报告、Doppler对照与原始覆盖表 |
| `navigation_demo` / `public_gsdc` | 已保存的历史模拟/GSDC解析样例                              |
| `learning_demo` / `learning`      | 独立学习演示和小练习，保留主实验                           |

`report.json`记录数据SHA、划分、特征、阈值和指标；`predictions.csv`存预测；`models.joblib`存可信本地产物。真实实验另保留冻结计划、数据分母与方法指纹，学习读取完整观测历史后再按参考人口评分。

对外发布经核验汇总至 [docs/experiments](../docs/experiments)，原始轨迹、权重、缓存和API连接不提交。已完成冻结目录拒绝覆盖，输入/实现变化用新目录。运行 [review_project.py](../scripts/review_project.py) 与 [review_positioning.py](../scripts/review_positioning.py) 仅检查/回放，不训练。
