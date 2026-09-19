# Paper RAG research library

这是一个由宿主 Agent 使用的论文 RAG 工具，不是单纯的 Wiki。它保留完整的
资料导入、混合检索、原文阅读、引用关系和证据定位能力；Wiki 只是可选的研究
成果层。普通查询只读，只有用户要求整理时才写入 Wiki。

一次研究请求的实际路径是：

`PDF/Markdown → MinerU/结构化资料 → SQLite FTS5 + Milvus → search → read → Agent 回答`

其中 `search` 负责召回候选证据，`read` 负责读取完整原文和上下文，宿主 Agent
负责问题拆解、补查、比较和最终回答。项目本身不调用生成式 LLM。

由宿主 Agent 理解问题、规划检索和撰写答案。本项目只负责保存资料、返回可追溯原文、检索和维护 Markdown Wiki，不调用生成式 LLM。

环境为 Conda `RAG_project`、Python 3.12。PDF 使用 MinerU，向量使用 Milvus/Zilliz，目录、版本、全文检索和任务使用 SQLite。Embedding 默认是 `qwen3.7-text-embedding-flash`，1024 维。

```powershell
conda activate RAG_project
python -m pip install -e .
paper-rag doctor --request request.json --json
```

`request.json` 是 UTF-8 JSON，检查状态时内容为 `{}`；`--request -` 从标准输入读取。参数名称使用 ASCII，中文问题和 Unicode 路径写在 JSON 中。未安装入口时可用 `python -X utf8 -m paper_rag`。复制 `.env.example` 配置所需服务，本地 `.env` 不进入 Git。仓库中的 `data/` 默认忽略；迁移、评测和 Wiki 结果属于本机资料资产。

| 操作 | 命令 | JSON 示例 |
| --- | --- | --- |
| 环境与一致性 | `doctor` / `status` | `{"remote":true}` |
| 导入旧资料 | `ingest` | `{"migrate_legacy":true}` |
| 导入 PDF / 笔记 | `ingest` | `{"sources":[{"path":"data/pdf/paper.pdf"}]}` |
| 增量向量索引 | `index` | `{}` |
| 独立重建 | `index` | `{"rebuild":true}` |
| 查论文、准确计数 | `papers find` / `papers count` | `{"query":"BERT"}` / `{}` |
| 混合检索 | `search` | `{"query":"attention mechanism","mode":"hybrid"}` |
| 读取原文 | `read` | `{"evidence_id":"ev:d_...:r_...:b000001:0:80"}` |
| 章节导航 | `read` | `{"document_id":"d_...","list_sections":true}` |
| 本地引用关系 | `citations` | `{"document_id":"d_...","direction":"outgoing"}` |
| Wiki 查询 | `wiki search` / `wiki read` | `{"query":"attention"}` / `{"page_id":"scaled-attention"}` |
| Wiki 任务与校验 | `wiki prepare` / `wiki lint` | `{}` |
| 中断发布恢复 | `wiki recover` | `{"page_id":"page-id"}` |
| 证据评测 | `eval` | `{"cases_path":"data/library/eval/reviewed-facts.json","mode":"lexical"}` |

完整请求模型位于 `paper_rag/library/contracts.py`。返回包含 `schema_version`、`status`、`data`、`warnings`。搜索与阅读使用 `next_cursor`，目录、引用和 Wiki 使用 `next_offset`；沿用原请求继续读取。`read` 可以指定 `context_before` / `context_after` 取相邻 block。stdout 只输出 JSON，依赖诊断写 stderr。

两份仓库内 Skill 位于 `.agents/skills/paper-research` 和 `.agents/skills/paper-curate`，包含按需加载的命令和发布规范。普通查询不写 Wiki；用户要求整理时，Agent 阅读原文、编写内容，再调用 `wiki apply`。机械校验通过不等于语义正确。

原文与解析产物保存在 `data/library/objects`；已发布引用使用文档、版本、block 和字符范围，重新切块不会使旧引用指向新文本。新版本先写入并核对向量，再发布。模型、服务部署或维度变化要求独立 generation；切换 Key 本身不改变向量身份。`index` 的 `rollback_generation` 可回到保留的完整 generation，旧数据和旧 collection 不自动删除。

首版边界：单机、个人使用、PDF 和 Markdown；无 Web、MCP、图数据库或独立回答服务。`ask/chat/plan` 已替换为清晰的宿主 Agent 迁移提示。原有解析与元数据适配保留用于兼容旧语料；旧生成、路由、翻译及旧关键词评测实现已删除，基线位于 `data/baselines/20260919T051652Z`。

运行 `python -m pytest -q` 进行离线验收。真实数据、质量评测、规模测试及未完成项记录于 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
