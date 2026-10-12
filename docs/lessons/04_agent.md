# 第四课：让回答引用实际证据

先使用离线问答查看 `answer`、`evidence` 与 `citation_check`。在已有真实产物的项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m navdiag ask "比较定位精度" --provider offline --artifacts artifacts/urbannav_positioning --task-context position_error
```

新克隆环境先按 [README](../../README.md) 生成学习演示，再把路径换为 `artifacts/learning_demo`。`offline`不请求问答模型。

## 三种回答方式

| 方式      | 谁选择证据                           | 是否调用LLM                      |
| --------- | ------------------------------------ | -------------------------------- |
| 离线速查  | 本地规则与检索器                     | 否，模板组织结果                 |
| 证据问答  | 本地先读取报告和相关片段             | 有依据时一次聊天请求，由模型解释 |
| 工具Agent | 模型从限定schema选择工具并可改写查询 | 多轮工具循环，受轮数/预算限制    |

有API不等于一定由模型选择工具。证据问答适合不支持function calling的模型；工具Agent要求服务正确支持工具调用与结果消息。

## 跟踪一个工具循环

阅读 [agent.py](../../src/navdiag/agent.py) 的 `EvidenceTools.definitions`、`invoke` 和 `ask`。模型请求例如 `compare_positioning({})`；程序校验工具名/参数，读取保存的报告并分配E1；结果返回模型，答案引用 `[E1]`。工具范围不含任意路径读取、执行命令或写实验。

`compare_nlos`读取卫星识别；`compare_models`读取位置告警；`compare_positioning`读取位置误差、覆盖和方法来源。当前数值以工具为准，文献摘要用于说明作者方法。`trip_summary`与`segment_summary`只读取观测/预测，不看离线参考标签。

网页主入口使用自选Chat Completions兼容服务。历史CLI其他协议适配仍保留，供应商、模型与是否支持工具取决于实际配置。配置步骤见[接口说明](../LLM_API.md)。

## 检索和引用

当前使用本地BGE与BM25/字符TF-IDF混合检索11条笔记、10份原创项目文档及20篇研究摘要；未准备编码器则明确回退关键词版。每次只取少量相关片段，保留文档、章节、行号和阅读范围，见[第七课](07_rag.md)。应用没有联网搜索工具。

多个文档可能都在E2检索结果中，它们应引用 `[E2]`，不能把L14文档ID变成 `[E14]`。`citation_check`只验证引用编号是否存在，不能判断正文是否被依据支持。

## 怎样检查回答

39个离线回归问题检查工具路由、知识来源和引用关系，不是答案正确率。真实GLM-5.3已完成17次证据问答，曾出现类别混淆、过度推断和引用问题；阅读[保留的失败与复测](../experiments/glm_quality.md)，不要把接口成功当作质量验证。

练习：分别跟踪一个模型比较、一个HDOP概念和一个无关问题。检查有没有附带不相关的指标、是否区分NLOS与位置告警、引用能否支持每个关键结论。能解释“哪些由本地代码完成，哪些交给LLM”，再进入RAG实现。
