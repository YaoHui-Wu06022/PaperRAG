# 本地论文库 RAG：当前实现与运行流程

本文描述 `new` 分支当前代码行为。系统将 ArXiv 论文、MinerU 解析、SQLite 元数据和引用关系、LlamaIndex 检索及 Milvus 向量连接起来，以 MCP 工具供客户端 Agent 使用。Agent 决定顶层工具并生成正文答案；服务端负责结构化数据、证据和确定性展示，不调用最终答案生成模型。

## 1. 文档导航

| 文档 | 内容 |
|---|---|
| [mcp.md](mcp.md) | 工具清单、参数、工具组、配置、确认边界及中文 CLI 调试 |
| [arxiv_acquisition.md](arxiv_acquisition.md) | 最新版本发现、PDF 下载、目录更新与失败处理 |
| [mineru.md](mineru.md) | 签名上传、任务轮询、结果校验和解析复用 |
| [chunk.md](chunk.md) | 区域、章节、正文及媒体切分、JSON、Catalog 和 FTS |
| [retrieval.md](retrieval.md) | JEV、关键词抽取、翻译、两路召回、四类任务和 Agent 回答 |
| [citation.md](citation.md) | 本地混合引用匹配、去重、方向、两跳和答案模板 |
| [index.md](index.md) | Embedding 缓存、全量与增量、Manifest 和 stale 原因 |
| [record.md](record.md) | 已执行的真实 MCP 请求、响应与 Agent 中文回答 |

说明页中的请求示例用于解释接口，不代表本次实际服务回包。真实输出以 record.md 的执行记录为准；论文数量、引用数量和排名会随入库与在线服务变化，不写成永久固定指标。

## 2. 数据和查询架构

```text
ArXiv ID / URL
  → Atom 元数据 + 最新版本 PDF
  → MinerU content_list.json / full.md / 图片
  → Catalog 同步
      ├─ papers / papers_fts：论文身份、元数据与发现
      ├─ chunks / chunks_fts：正文结构与 BM25
      ├─ references / citation_edges：本地引用匹配与有向图
      ├─ mineru/chunks.json：逐篇人工检查副本
      └─ citation_graph.json：全库引用检查副本
  → 单独执行索引同步
      → TextNode（使用现有 Chunk）
      → DashScope Embedding
      → Milvus + Manifest + SQLite Embedding 缓存

用户中文问题 → 客户端 Agent 选择 MCP 工具
  ├─ library_search → 元数据 BM25 / SQL → verbatim 模板
  ├─ library_citation → SQLite 图 / 引用条目 → verbatim 模板
  ├─ library_retrieve → JEV + 改写/翻译 + BM25/向量 + 任务证据 → compose
  └─ library_read / library_get_chunk → 原文与完整来源
```

SQLite 是论文记录、Chunk 和引用关系的查询来源；原始 PDF、MinerU 文件是追溯来源，JSON 是查看副本。Milvus 保存正文向量，不作为引用图数据库。LlamaIndex 使用已有 Chunk 构造 TextNode，并通过 VectorStoreIndex/MilvusVectorStore 实现语义召回，不重新解析 PDF 或切块。

## 3. 模块职责

| 实现位置 | 当前职责 |
|---|---|
| `paper_rag/config.py` | 读取根目录 .env，进程环境覆盖；路径和服务参数 |
| `paper_rag/http.py` | JSON HTTP、重试、响应解码及错误封装 |
| `paper_rag/acquisition/arxiv.py` | ID/URL、Atom 元数据、最新版本判断、PDF 和资产 manifest |
| `paper_rag/ingest/mineru.py` | 本地 PDF 提交、签名上传、轮询、ZIP 校验和落盘 |
| `paper_rag/catalog/chunks.py` | 区域和章节树、边界切分、表格检索文本、媒体上下文 |
| `paper_rag/catalog/service.py` | SQLite schema、SQL 过滤、Catalog 重建、Chunk/图查询与 JSON 导出 |
| `paper_rag/catalog/references.py` | 引用条目解析、身份规范化、混合匹配和重复条目标记 |
| `paper_rag/routing/` | RETRIEVE 下的四类正文任务；JEV 失败时本地规则 |
| `paper_rag/llamaindex/query_rewriter.py` | OpenAI 兼容接口，只抽取 entities/core_terms |
| `paper_rag/llamaindex/translation.py` | BM25 查询准备；腾讯云主翻译、阿里云备用 |
| `paper_rag/llamaindex/nodes.py`、`embedding.py`、`index.py` | 节点、Embedding、Milvus/缓存/Manifest 生命周期 |
| `paper_rag/llamaindex/retrievers.py` | 词法、向量、Chunk 去重、RRF 和编号排序 |
| `paper_rag/llamaindex/service.py` | 输入、硬约束、目标发现、四类证据组织和统一响应 |
| `paper_rag/reading/service.py` | full.md 分页和 Chunk 读取 |
| `paper_rag/presentation.py` | 元数据/引用答案模板；正文任务的 Agent 指令 |
| `paper_rag/mcp/`、`paper_rag/cli/` | 服务入口、工具注册、作业和 CLI handler |

## 4. 三类问题的边界

元数据问题，例如“2020 年以后有哪些计算机视觉论文和注意力相关”，由 Agent 调用 library_search 并传年份/分类 filters。非空 query 经过双字段抽取、必要的翻译和 papers_fts 检索；query 为空时只做 SQL 过滤。不读取正文 Chunk，不调用 JEV 或向量。标题和作者来自 SQLite；返回记录时只对该论文资产做存在检查，不重新扫描全库。

引用问题，例如“Attention Is All You Need 的引用和被引用关系”，直接调用 library_citation，可传 paper_title。它从本地 references/citation_edges 查询，both 默认两跳；out/in 默认一跳。多跳引用沿箭头同向走，多跳被引用沿箭头反向走，不能把共享参考文献当作间接引用。

正文问题只调用 library_retrieve。默认 task=auto、mode=hybrid、limit=8。Agent 选择工具后，JEV 才将问题分类为 fact/reason/summary/comparison；JEV 不选择元数据或引用工具。Query Rewriter 和翻译为 BM25 准备词，向量仍使用原始问题。各任务分别选择直接证据、邻域窗口、章节覆盖或逐篇均衡证据。

paper_ids 和 filters 是正文硬约束，取交集并同时限制两路；fact/reason 不会先用词法命中的论文集合砍掉向量范围。默认检索 abstract/content；提到附录或显式指定 regions 才加入 appendix。Reference 永不进入正文索引和召回。

## 5. 展示原文、检索表示与回答证据

- Chunk.text 保留展示原文，包括表格 HTML 和公式 LaTeX。
- Chunk.retrieval_text 加入完整章节路径及媒体上下文，表格转换为结构化纯文本；paper_id 不加入文本前缀。
- chunks_fts 除 retrieval_text 外还索引章节路径、区域、标题；TextNode.text/Embedding 使用 retrieval_text。
- data.evidence 是最终回答证据卡片，仅保留正文、基本定位与 score；BM25/向量排名位于 retrieval_debug。完整资源、检索文本和来源 Block 可按 chunk_id 获取。
- source_id 是单次最终响应中的 S1、S2 等编号；chunk_id 是持久来源键。reason 窗口合并时 source_chunk_ids 记录实际组成。

search/citation 返回 render_policy=verbatim 和非空 answer_text，Agent 原样输出。正文返回 compose、空 answer_text、当前任务的 agent_instruction，由 Agent 根据 evidence.text 整合中文回答并使用真实 [S#]。MCP 不自动给宿主注入系统消息，也不提供答案生成或答案校验工具。

## 6. 更新流程与一致性边界

下载、解析、Catalog 和向量同步是独立阶段，没有自动串行全链路写入。按需执行：

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag acquire arxiv 2106.09685 --dry-run --json
conda run -n RAG_project python -X utf8 -m paper_rag acquire arxiv 2106.09685 --json
conda run -n RAG_project python -X utf8 -m paper_rag ingest arxiv 2106.09685 --dry-run --json
conda run -n RAG_project python -X utf8 -m paper_rag ingest arxiv 2106.09685 --json
conda run -n RAG_project python -X utf8 -m paper_rag catalog sync --json
conda run -n RAG_project python -X utf8 -m paper_rag catalog chunks --paper-id 2106.09685 --json
conda run -n RAG_project python -X utf8 -m paper_rag index status --json
conda run -n RAG_project python -X utf8 -m paper_rag index rebuild --mode auto --json
```

MCP 下载、解析、索引同步需要 confirm=true 并提交异步作业；Catalog 同步确认后直接同步执行。CLI 写命令同步执行，不要求 confirm。配置与中文查询调试见 mcp.md。

当前失败恢复能力必须分别理解：

| 阶段 | 行为和边界 |
|---|---|
| ArXiv/MinerU | 临时输出校验后目录替换，常规失败保留已有结果；新 PDF 版本需重新解析 |
| Catalog | 临时 SQLite 和 JSON 准备完成后逐个替换；每个文件原子，不是多文件事务 |
| full 向量同步 | 临时 Collection 验证后更新缓存和 Manifest，切换前常规失败保留旧索引 |
| incremental 向量同步 | active Collection 原地 upsert/delete；缓存可恢复，已写向量不完整回滚 |

引用匹配或回答模板改变不要求重新 Embedding；Catalog 同步刷新时间戳可能导致 stale，可用 auto 更新状态。Chunk 规则、模型或维度改变触发全量边界。详见 index.md，不能把全部索引模式都称为临时 Collection 原子切换。

## 7. 当前限制与验证

公开 max_chars 参数当前未参与证据截断；limit 是最多返回的证据条数，reason 合并后可能更少。没有自动将工具指令变成宿主 system 消息；最终事实是否被证据支持仍由 Agent 负责。引用匹配是本地启发式高置信匹配，不保证召回所有真实引用；图 filters 的端点规则也不同于正文硬约束。

索引首次加载涉及 Milvus 初始化，通常耗时较高。长期运行 MCP 缓存配置和 IndexService；外部修改 .env 或更新索引后应考虑重启，以实际状态和调试字段判断是否发生降级。

本次代码提交前在 RAG_project 验证：159 项测试通过，pip check 无依赖冲突。运行方式：

```powershell
conda run -n RAG_project python -X utf8 -m pytest -q
conda run -n RAG_project python -X utf8 -m pip check
```

真实回归保存在 record.md，使用中文问题、真实 MCP 请求和宿主 Agent 回答；不使用模拟答案脚本，不调用 Query Rewriter 模型生成最终答案。本轮文档更新未同步 Catalog 或重建向量。
