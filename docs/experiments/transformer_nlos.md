# 单卫星因果 Transformer 对比（2026-10-11）

仅在 D:/SHIXI/navdiag 本地开发，未上传 GitHub；MMTNet 暂不接入。当前比较完整分类器，移除两项特征 RF 消融。数值快照见 [JSON 报告](transformer_nlos_report.json)，初期数据来源记录见 [v0.3](v0_3_nlos.md)。

## 固定实验设置

沿用 KLT 已清洗五项特征和区域隔离划分：KLT2 两段训练 9,830 行，KLT1 验证 6,607 行，KLT3 测试 10,085 行。数据 SHA-256 为 251e8ecda9f5a80ed65a76b67e2eec495d03a6618947e3a812880198ff46a1f0。只有一天、一个接收机、三个区域，行之间有时间和同历元相关性。

Transformer 对同一 trip/device/satellite 的过去及当前 32 次观测分类，超过 5 秒间隔重置；不足 32 项时在左侧重复首条已知观测。32 次不是固定 32 秒。因果注意力屏蔽未来，最终位置的表示输出 NLOS logit。标签、相机、物体类型、分组 ID 不进入特征；未知标签观测可提供历史，但不计算监督损失。推理调用方要提供历史；逐行独立调用无法利用历史。

网络使用 [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)，两层、四头、宽度 64、前馈 128、dropout 0.1、正弦位置编码，共 67,521 个参数。独立初始化每层；训练集拟合中位数和 StandardScaler。seed=42，CPU PyTorch 2.14.1+cpu，两线程；类别加权 BCE、AdamW、lr=0.0005、weight_decay=0.01、batch=128、梯度裁剪 5。

最大 30 轮，验证加权 BCE 早停 patience=6，实际运行 7 轮，恢复第 1 轮参数。之后在验证集 F1 的 0.05–0.95 网格上选阈值 0.65。未尝试根据测试效果改变网络/窗口/轮数。LR/RF 使用当前五项特征，Transformer 还使用历史，因此这是完整方法对比，不能归因成纯架构收益。

## 结果与选模

| 方法 | Train F1 | Val Precision | Val Recall | Val F1 | Test Precision | Test Recall | Test F1 | 测试误报 / 漏报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.741 | 0.807 | 0.694 | 0.746 | 0.796 | 0.658 | 0.720 | 609 / 1,232 |
| RF（验证集选中） | 0.853 | 0.708 | 0.812 | 0.757 | 0.672 | 0.778 | 0.721 | 1,366 / 801 |
| Transformer | 0.768 | 0.779 | 0.714 | 0.745 | 0.782 | 0.653 | 0.712 | 658 / 1,249 |
| C/N0 固定规则 | — | 0.663 | 0.639 | 0.651 | 0.621 | 0.625 | 0.623 | 1,377 / 1,352 |

用户希望同时考虑 Precision 与 NLOS 判别正确率，不设硬性 90% 召回门槛。这里将后者按实际 NLOS 中正确检出的比例（Recall）解释，F1 为二者调和平均。模型和阈值按验证集选择，测试只报告效果。RF 是当前实验模型；尚未确定最终使用成本和最终部署模型。若“NLOS 判别正确率”指全部 LOS/NLOS 的总体准确率，网页另列 Accuracy，避免混同。

RF 的召回更高，Precision 较低；LR/Transformer 报警更少，也漏掉更多 NLOS。只有相近 F1 不能证明三种方法等效，更不能支持 Transformer 改善定位的结论。当前测试区域已在开发中查看过；进一步调优需要新的未查看最终留出集。

## 效果一般的原因与可支持的判断

RF 的训练/测试 F1 差距约 0.131，表明训练拟合不能充分转移至留出区域，但不能单凭此区分过拟合和区域分布差异。单日少量短采集缺少足够独立场景；强信号可能来自 NLOS 反射，弱信号也可能是 LOS。当前五项特征无法消除这种混淆，物体类型和未跟踪卫星真值也缺失。相机标签和上游特征提取没有独立重做。

Transformer 最佳验证损失在第 1 轮，后续训练未持续改善验证损失；简单扩大网络或延长训练没有数据依据。测试 AUC 为 0.871，RF 为 0.866，但阈值下 Transformer F1 较低，不可只挑更好的指标宣传。当前实验优先暴露这些限制。

## 运行与代码学习

```powershell
.\.venv\Scripts\python.exe -m navdiag train-nlos --with-transformer --transformer-epochs 30
.\.venv\Scripts\python.exe -m navdiag serve --artifacts artifacts/navigation_demo --nlos-artifacts artifacts/klt_nlos
```

源码为 `src/navdiag/nlos_transformer.py`，集成在 `src/navdiag/nlos.py`。测试覆盖卫星/设备/行程/间隙隔离、未来观测与注意力泄漏、缺少历史时的有限输出、模型保存加载与乱序推理、训练集填充参数，以及修改测试标签不改变训练权重/验证阈值/选中模型。
