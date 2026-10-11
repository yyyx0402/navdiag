# 开源参考与复用记录

用户授权参考其他开源项目完善 NavDiag。当前复用已有数据、特征表和成熟库，把实现重点放在标签、无泄漏评估、错误分析与导航验证。

| 来源 | 本项目实际使用方式 | 许可与边界 |
| --- | --- | --- |
| [KLTDataset](https://github.com/ebhrz/KLTDataset) | 九段可靠标签采集、三个日期；下载五段新增原始标签、RINEX 与广播星历；按时间戳及卫星 ID 核查 | 上游 GPL-3.0；原始数据留在本地，排除原作者声明不可靠的 KLT3-649 |
| [UrbanNav](https://github.com/IPNL-POLYU/UrbanNavDataset) | 三城区 F9P splitter 观测、SPAN 参考位置和静态建筑图；生成独立几何参考并执行新城区测试 | README 没有明确数据再分发许可；原始/派生表与模型留在本地，不声称人工 NLOS 真值或跨城市验证 |
| [NOAA](https://ngs.noaa.gov/CORS/data.shtml) / [SatRef](https://www.geodetic.gov.hk/en/rinex/downv.aspx) | 获取三次 May 2021 采集当天 GPS / 北斗广播星历 | SatRef 个人非商业内部使用限制，文件不再分发；来源和 SHA 固定在 UrbanNav 清单 |
| [Fossett Laboratory gnss-nlos-notebook](https://github.com/fossettlab/gnss-nlos-notebook) | 固定四段数值表提供旧段标签；本地缓存并调用固定版本提取器，统一重算九段特征；旧段数值全部复核一致 | 固定 b13aa856f723a472bd35d45171017580642054cf；GPL-3.0；上游源码在忽略的缓存，独立适配工具在 tools/klt，使用单独 GPL 许可证，不打包进主 Python 包 |
| [Stanford gnss_lib_py](https://github.com/Stanford-NavLab/gnss_lib_py) | 使用公开 GSDC 解析测试文件，参考字段与教程 | 未声称复现完整定位算法；数据条款单独适用 |
| [Stanford deep_gnss](https://github.com/Stanford-NavLab/deep_gnss) | 下载两个公开 GSDC 观测/参考例子 | 用于历元定位失准流程，缺少 NLOS 真值 |
| [scikit-learn](https://scikit-learn.org/stable/) | 直接使用 Pipeline、填充、LR、RF 与评估实现 | 项目依赖；不重新实现通用算法 |
| [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html) | 直接使用注意力/前馈层；本项目实现单卫星因果窗口、位置编码、训练与评估 | 库依赖；没有复制 MMTNet 或其他研究项目源码 |
| [DeepNLOSDetection](https://github.com/rwth-irt/DeepNLOSDetection)及[论文](https://arxiv.org/abs/2309.00480) | 参考卫星级任务、天空掩膜、特征与域外评估思想 | 核查时仓库仅有说明/展示资料；未获取训练代码或数据，不声称完成复现 |

本项目主 Python 包使用 MIT 许可。[独立 KLT 预处理工具](../tools/klt/README.md)使用 GPL-3.0-or-later。上游 GPL 数值数据、其派生表和训练产物留在本地忽略目录，不作为 MIT 原创数据打包发布。以后分发上游源码、衍生实现、数据或模型时，需保留相应许可和署名并核对具体分发条件。

[UrbanNav 独立预处理工具](../tools/urbannav/README.md)同样调用固定版本 GPL 提取器，使用 GPL-3.0-or-later，位于主 Python 包以外。独立几何参考计算与实验执行代码没有复制上游实现；标签来源、未知覆盖和地理范围见[新城区实验报告](experiments/urbannav_generalization.md)。

KLT 数据引用：Hu, R., Wen, W., & Hsu, L.-T. (2023). *Fisheye camera aided NLOS exclusion and learning-based pseudorange correction*. IEEE ITSC, 6088–6095。正式引用与版本以原发布页为准。

数值清洗：源表 927 条 C/N0=0 记录按缺测保留；历史 C/N0 标准差由本地适配器重算，五个过去/当前观测、至少两条有效值、超过 5 秒间隔重置。原始特征表仍保持下载时的 SHA-256。
