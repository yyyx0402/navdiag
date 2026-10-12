# 文档

| 内容 | 入口 |
| --- | --- |
| 数据、字段与许可 | [数据说明](../data/README.md)、[来源与复用](OPEN_SOURCE_REFERENCES.md) |
| 任务、特征、模型与限制 | [模型卡](MODEL_CARD.md)、[标签](OBSTRUCTION_LABELS.md) |
| 当前识别结果 | [KLT 双协议](experiments/nlos_protocols.md)、[UrbanNav 城区留出](experiments/urbannav_generalization.md)、[分布审计](experiments/nlos_domain_audit.md) |
| 当前定位结果 | [完整路线评估](experiments/positioning_expansion.md)、[方法来源](research/positioning_methods.md) |
| 问答与文献 | [API](LLM_API.md)、[RAG](RAG_RETRIEVAL.md)、[诊断质量](experiments/agent_quality.md)、[历史回答检查](experiments/glm_quality.md)、[文献目录](research/reading_list.md) |
| 技术背景 | [NLOS](knowledge/nlos.md)、[泛化](knowledge/generalization.md)、[Zhang 2023 方法对照](research/zhang2023_reproduction.md) |

## 可选依赖

在项目虚拟环境安装所需组件，PyTorch 可选 CPU 版本：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[temporal,gnss,rag]"
.\.venv\Scripts\python.exe scripts/prepare_semantic_index.py --download
```

最后一条下载固定版本 BGE 权重，在本机编码。没有权重时可用关键词检索。真实数据复现入口为 [KLT 工具](../tools/klt/README.md)、[UrbanNav 工具](../tools/urbannav/README.md)和各结果页；下载量较大，需遵循来源条款。

## 结果复查

```powershell
.\.venv\Scripts\python.exe scripts/review_project.py
.\.venv\Scripts\python.exe scripts/review_positioning.py
```

这两项读取已完成的本地产物并检查一致性或回放，不训练。冻结目录禁止覆盖，输入或实现变更需使用新目录。远程 CI 检查代码与模拟流程，不下载完整真实数据重跑。
