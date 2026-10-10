# 诊断 Agent 的实现与验证

## 三层流程

1. 算法层从观测产生报告、预测和定位对照，保存不可随问答修改的实验产物。
2. 工具层读取限定实验目录，返回结构化结果与 `[E#]` 证据编号。本地 TF-IDF 检索八篇原创笔记，保留原始参考网址。
3. 回答层选择 offline / openai / deepseek / compatible。offline 采用简单规则和模板；API 模式实际把工具 schema、函数调用结果送回模型，直到获得最终回答或用尽预算。

`trip_summary` / `segment_summary` 只读取观测和预测。`compare_models` / `compare_positioning` 是明确标记的离线参考评测。允许工具不包含任意读文件、执行命令或写入能力。

## 本地配置

复制 `.env.example` 到 `.env`，取消一个后端配置段的注释，填写该后端的 key。也可以使用环境变量 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`。通用 `NAVDIAG_API_KEY` 只属于其声明的 `NAVDIAG_PROVIDER`，网页切换后端时不会把这个 key 转发给另一个后端。

OpenAI 使用 `/responses`，默认 `gpt-6-luna`，请求 `store=false`；DeepSeek 使用 `/chat/completions`，默认 `deepseek-flash`，当前适配关闭 thinking。具体模型可调整为账户支持且具备工具调用能力的模型。API 问题和证据会发送给所选供应商；仅在网页点击分析或 CLI 调用时发起请求。

工具循环最多 5 轮、每轮 6 个调用，每个响应输出上限 1,500 tokens，默认连接超时 60 秒。没有 key 时 API 模式给出错误，不自动回退并假装已调用模型。

接口依据：[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)、[OpenAI 模型文档](https://developers.openai.com/api/docs/models/gpt-6-luna)、[DeepSeek Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion/)。模型名称是本版本配置，不保证未来永远可用。

## 固定评测

`resources/agent_cases.json` 的 30 个问题覆盖模型/定位/数据/片段/技术概念。`evaluate-agent` 逐题保存回答、工具证据、usage、延迟和以下检查：

- 是否调用预期工具。
- 是否引用已存在的证据 ID，是否包含伪造 ID。
- 工具证据是否包含真实/模拟来源标记。
- 指定问题的目标知识条目是否进入前三条检索结果。

在当前模拟与真实实验上，offline 均通过 30/30。API 的两种协议用模拟供应商响应验证；没有配置 key，尚未运行真实模型评测。30/30 不能解释为答案事实正确率、根因推理正确率或大模型能力。

要报告实际 LLM 成绩，需要 `--provider openai` 或 `deepseek` 运行同一题集，再人工核对数字对应的指标/单位/划分，数据边界、未知结论和用户问题是否被回答。只检查 `[E1]` 存在并不能证明正文每个数字受证据支持。
