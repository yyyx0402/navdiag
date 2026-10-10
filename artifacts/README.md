# Experiment artifacts

实验产物保存在本目录的子目录中，由 Git 忽略。先运行 `python -m navdiag demo`。

`report.json` 保存数据 SHA-256、行程划分、随机种子、scikit-learn 版本、特征和指标。需要对外展示时，选择经过核验的指标表和图加入 `docs/experiments/`，同时说明数据来源、参数、划分和局限；不上传原始轨迹或模型缓存。
