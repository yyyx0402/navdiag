# 第三课：从基线到五种神经网络

先读 [nlos.py](../../src/navdiag/nlos.py) 的 `build_classic_model`，再读 [learning.py](../../src/navdiag/learning.py) 的 `evaluate` 和 `choose_threshold`。两种任务复用指标函数，但模型、特征和标签分别定义。

## 训练、验证和推理

`fit`学习训练参数；`predict_proba`产生评分。Pipeline将填充、标准化和分类器组合，验证/测试只使用训练保存的预处理参数。NLOS RF默认200树、深度12、最小叶8；历元告警RF为150树、深度8、最小叶6。

验证数据选择阈值和模型；神经网络还使用验证BCE选择早停检查点。当前完整实验的选模指标为验证F1。测试Precision/Recall用于描述结果，不能看完测试再挑阈值或部署模型。改变业务目标时需预先声明目标，再用开发验证数据决策。

## 误报与漏报

矩阵为 `[[TN, FP], [FN, TP]]`，Precision=`TP/(TP+FP)`，Recall=`TP/(TP+FN)`，F1为两者的调和平均。NLOS任务的正类是NLOS；位置告警的正类是基线误差超过阈值。

没有真正正样本时，Recall分母为0，无法评估漏报。代码报告null，不把全预测正常当作完美检测。原GSDC 11点留出就属于这种历史解析样例；新的UrbanNav告警测试包含正样本。

评分尚未做概率校准，0.8不等于已经验证的80%失准概率。改变阈值不改变原始评分排序；需要同时检查误报数、漏报数和使用场景，而不只看一个F1。

## 因果历史与五网络

NLOS五网络共用同一卫星的32次过去/当前观测、训练集拟合预处理和固定预算。阅读 [nlos_transformer.py](../../src/navdiag/nlos_transformer.py) 的 `satellite_windows` 与 [nlos_neural.py](../../src/navdiag/nlos_neural.py) 的 `build_network`：

| 模型        | 怎样使用历史                           |
| ----------- | -------------------------------------- |
| MLP         | 展开窗口，再通过全连接层               |
| 1D CNN      | 用卷积提取局部时间关系                 |
| LSTM / GRU  | 单向更新历史状态                       |
| Transformer | 使用因果注意力组合窗口信息，分类当前项 |

历史按采集段、设备、卫星分组，间隔大于5秒时重置。短历史重复首个可用观测补齐；32次不一定是32秒。`unknown`可提供观测历史但不提供监督。LR/RF只有当前输入，因此网络与树模型对比也包含历史访问差异。

位置告警 [temporal.py](../../src/navdiag/temporal.py) 的GRU则用10历元、六项特征，别与32次卫星窗口混用。新的定位流程保留全部成功观测的历史，合格参考只选择监督/评分行。

## 用已有报告判断效果

旧四段Transformer的F1 0.712是[历史结果](../experiments/transformer_nlos.md)。当前应分别读取 [KLT双协议](../experiments/nlos_protocols.md)、[UrbanNav迁移/城区互斥](../experiments/urbannav_generalization.md)与[位置告警](../experiments/positioning_expansion.md)。先核对任务、数据、分母和验证选择，再比较排名。

当前Transformer只处理同一卫星的历史，完整Zhang跨卫星结构尚未复现。强模型名称不保证更高成绩，一次差异也不能证明稳定优势。

## 练习与验收

1. 手算 `TP=8, FP=2, FN=4, TN=6` 的三个指标。
2. 用十条合成标签/评分对比0.3与0.7阈值，解释哪些样本改变预测。
3. 画出前三次观测的窗口，再读 [因果测试](../../tests/test_nlos_transformer.py)，解释未来观测为何不能改变过去窗口。
4. 阅读一个已保存折的验证阈值与测试混淆矩阵，解释所选模型为何不必是测试最高模型。

`models.joblib`只加载自己生成或可信的文件。推理需要完整的观察历史及分组时间，参考误差无需提供；删掉无参考点会改变窗口，所以应在观测表上推理，再按参考人口评分。
