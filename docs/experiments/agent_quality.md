# 诊断助手质量检查

2026-10-12，使用同一GLM-5.3连接、保存的2048输出tokens/default推理/90秒超时，核对证据问答与真实多轮工具调用。未修改连接配置。

## 输入与结果

| 同一问题 | 原输入tokens | 首轮精简后 | 变化 |
| --- | ---: | ---: | ---: |
| UrbanNav RF/Transformer两协议对比 | 17797 | 5291 | −70.3% |
| LSTM训练设置 | 5854 | 3031 | −48.2% |

输入由供应商usage记录；这是两题的工程对照，后续提示/资料继续修订。生成长度与耗时仍有随机性，对比题耗时26.4→32.1秒，设置题15.5→12.3秒；不能据此声称普遍提速。

首轮九道真实回答中五道通过开发审查，四道需核对；另有一题无关问题由本地保护处理、零模型请求。问题覆盖模型指标、训练设置、HDOP、unknown、论文复现、缺失城市、定位改善与两题工具调用。

修订后五题复测中三题通过，两题仍有过强几何表述或无依据文献扩展；针对这两题修订资料选择后，两题复测通过。失败与复测均保留。工具测试确认模型实际调用了工具并使用返回依据；限制任务工具后，城区比较不再读取历元告警。

开发助手审查，全部为已见问题，不是独立专家/盲测。不同轮次不能合并为最终版本正确率。数值核对、工具调用成功和完整事实正确分别评价，摘要见[JSON](agent_quality_report.json)。完整回答保存在本地忽略产物，不随仓库发布。

## 复现

先准备真实实验产物和本地模型连接。以下显式调用远程模型、可能消耗服务额度；自动检查工具、引用及有限的显式数值/方向，完整语义仍需逐题审阅：

```powershell
.\.venv\Scripts\python.exe -m navdiag evaluate-agent --connection 你的连接ID --answer-mode tools --cases src/navdiag/resources/agent_tool_cases.json --artifacts artifacts/urbannav_positioning --nlos-artifacts artifacts/klt_protocols --generalization-artifacts artifacts/urbannav_generalization --out artifacts/my_tool_check.json
```

真实测试通过不保证所有问题或重复生成稳定。一般证据问答用一次请求；工具模式受轮数限制，可能使用更多请求。概念词义、引用支持和未测因果仍需要人工核对。
