# C/N0 区域分布审计

KLT共享测试段上的固定规则得分相同是协议结果，不能证明区域分布相同。这里直接比较观测分布，非正C/N0缺测值剔除；LOS/NLOS条件统计仅使用参考合格标签。

| 区域 | 全部有效 C/N0 中位数 | 参考 LOS 中位数 | 参考 NLOS 中位数 |
|---|---:|---:|---:|
| KLT1 | 32 | 33 | 27 |
| KLT2 | 31 | 33 | 27 |
| KLT3 | 31 | 33 | 27 |
| TST | 34 | 36 | 23 |
| WP | 37 | 39 | 24 |
| MK | 36 | 43 | 27 |

![经验分布](../images/nlos-cn0-domain-audit.png)

KLT全体两两KS距离0.029–0.042，Wasserstein0.35–0.49 dB-Hz；按标签/星座/仰角匹配后加权距离0.87–0.92。UrbanNav对应全体KS0.207–0.265，Wasserstein1.48–2.74，条件加权距离2.26–5.57。当前同城三城区存在经验差异。

C/N0还受仰角、天线、安装、跟踪、日期和标签筛选影响，不能归因于城市本身。大量卫星行相关，单次区域采集不足以支持IID显著性或跨城市结论。旺角标签覆盖19.9%，条件分布受筛选影响明显。

[Zhang2023](https://arxiv.org/abs/2309.00480)将HK/AC分布差异讨论为泛化因素；[García Crespillo2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10416230/)提出开阔场景预归一化，需独立校准，不能用测试标签重新归一化。

统计与指纹见[JSON](nlos_domain_audit.json)。`python scripts/audit_nlos_domains.py` 读取本地处理表，不训练。KLT是小尺度迁移比较，UrbanNav是同城未见城区测试；跨城市仍需独立原始观测及NLOS参考。
