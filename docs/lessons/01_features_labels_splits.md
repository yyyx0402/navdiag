# 第一课：机器学习到底在学什么？

这节课先读 `src/navdiag/data.py`，不要求马上理解全部项目。

## 1. 用一句话定义任务

“根据接收机当前可获得的信息，预测这一时刻的水平定位误差是否大于 10 米。”

一行数据是一次定位时刻。输入 X 有 6 个数：卫星数量、载噪比均值和最小值、HDOP、伪距残差 RMS、速度。输出 y 是 0 或 1：

```python
y = (horizontal_error_m > 10).astype(int)
```

误差超过 10 米时，标签为 1；否则为 0。这个标签来自“基线定位结果与独立参考轨迹”的比较。真实运行时没有参考轨迹，模型只能看 X。

`FEATURES` 是明确的允许列表。`horizontal_error_m`、`trip_id`、`timestamp_ms`、`data_kind` 都不进入模型。特别是把真实误差作为输入，会让模型看见答案，离线结果再好也没有应用价值。

## 2. 专业知识怎样进入模型

载噪比描述信号强度相对噪声的情况。HDOP 描述卫星几何对水平位置精度的影响。伪距残差描述观测与基线解的拟合差异。

这些特征帮助模型判断风险，但单独一个特征不能证明“发生了 NLOS”。例如残差很小仍可能存在多个观测共同偏差，所以必须在真实数据上验证。

## 3. 为什么按整段行程划分？

相邻两个时刻的卫星、环境和运动状态很接近。如果把一条轨迹的奇数行训练、偶数行测试，模型可能只是遇到了几乎相同的场景。

我们先划分 `trip_id`，再选择对应数据行：

```python
frame[frame["trip_id"].isin(training_trip_ids)]
```

三个集合各自的职责是：

| 集合 | 用途 |
| --- | --- |
| 训练集 | 学习模型参数；标准化器也只在这里拟合 |
| 验证集 | 选择模型、参数与分类阈值 |
| 测试集 | 在方案确定后检查留出行程的表现 |

同一次驾驶的多部手机必须归在同一 `trip_id`。真实实验还要考虑重复路线和不同设备，让最终评测接近实际使用场景。

## 4. 运行后看这两处

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m navdiag demo
```

打开 `artifacts/demo/synthetic_epochs.csv`，核对每列含义。再打开 `report.json` 的 `splits`，确认 train、validation、test 没有共同的行程 ID。图里的 `SYNTHETIC` 表示当前结果来自模拟数据。

## 5. 三个小练习

1. 手选一个历元：如果水平误差是 8.4 米、阈值是 10 米，它的标签是多少？如果阈值改成 5 米呢？
2. 找到 `FEATURES`，解释为什么 `horizontal_error_m` 没有出现在里面。
3. 阅读 `split_by_trip`，用自己的话说明 `isin()` 如何把行程 ID 转成行选择。

你可以直接向我提问：“逐行讲解 data.py 的 split_by_trip”或“用一个小表格解释 X 和 y”。下一课再进入模型训练。

延伸阅读：[scikit-learn 官方文档：数据划分、验证集和预处理](https://scikit-learn.org/stable/modules/cross_validation.html)。
