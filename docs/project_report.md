# 本地论文库 RAG 项目整体报告

## 项目目标

本项目把 ArXiv 论文原文、MinerU 结构、章节和区域、表格/公式/图片、论文元数据、引用关系以及可回溯检索证据组织成完整 RAG 数据链路。

核心原则是：先解析和结构化，再执行多路召回；结果必须带 paper_id、chunk_id、章节、页码和 source_id；系统负责检索与证据组织，不直接生成最终答案。

## 文档导航

- [ArXiv 获取](arxiv_acquisition.md)
- [MinerU 解析与入库](mineru.md)
- [Chunk、Catalog 与索引](chunk.md)
- [正文检索](retrieval.md)
- [实际调用记录](record.md)

## 总体架构

索引链路：

    ArXiv ID/URL
      -> Atom 元数据和 PDF
      -> MinerU content_list.json
      -> SQLite Catalog / FTS5
      -> retrieval_text
      -> DashScope Embedding
      -> Milvus active collection

查询链路：

    用户问题
      -> library_search 或 library_retrieve
      -> filters / 候选论文发现
      -> lexical + semantic + RRF
      -> task 相关排序和证据窗口
      -> source_id、章节、页码和正文上下文

## 主要模块

- paper_rag/acquisition/arxiv.py：ArXiv 元数据和 PDF；
- paper_rag/acquisition/mineru.py：MinerU 远程解析和任务；
- paper_rag/catalog/chunks.py：区域、章节和 Chunk；
- paper_rag/catalog/service.py：Catalog 和 FTS5；
- paper_rag/llamaindex/nodes.py：SQLite Chunk 到 LlamaIndex Node；
- paper_rag/llamaindex/embedding.py：DashScope Embedding；
- paper_rag/llamaindex/index.py：Milvus 生命周期；
- paper_rag/llamaindex/retrievers.py：lexical、semantic、RRF 和质量排序；
- paper_rag/llamaindex/service.py：候选发现、任务证据组织和统一 retrieve；
- paper_rag/mcp/：MCP 工具和异步 Job。

## 当前状态

Catalog 包含 34 篇论文、3292 个 Chunk。区域计数为 abstract 58、content 2297、appendix 937，Reference 不生成正文 Chunk。

当前规则版本为 content-list-regions-v6-boundary-aware。Milvus active collection 已完成全量重建：

- indexed_count：3292；
- embedding_dimensions：1024；
- index_ready：true；
- index_stale：false；
- cache_complete：true。

## 数据契约

- text：原始展示和追溯文本；
- retrieval_text：SQLite FTS5 和 Embedding 的唯一输入；
- metadata：论文、章节、页码、区域、来源 block、资源引用和哈希；
- chunk_id：稳定的证据定位键；
- source_id：一次响应中的展示编号，不替代 chunk_id。

区域契约为 abstract、content、appendix、reference；reference 只进入引用图，不进入正文向量。

## 生命周期与一致性

Catalog 和 Milvus 都使用临时输出、校验和原子切换。规则版本、retrieval_text_hash、Embedding 模型和维度变化会让索引变旧，auto 模式会转全量重建。构建失败时保留旧 Catalog、cache 和 active collection。

## MCP 工具边界

- library_search：元数据搜索；
- library_retrieve：正文证据；
- library_get_metadata：论文元数据；
- library_read：全文或文件读取；
- library_citation：引用、被引用和局部图；
- library_index_status / library_index_rebuild：索引状态和重建；
- library_arxiv_download / library_arxiv_ingest：获取与 MinerU 任务。

Agent 识别规则保持最小化：`library_search` 用于元数据并直接输出 `data.presentation.answer_text`；`library_citation` 用于引用关系并直接输出 `data.presentation.answer_text`；`library_retrieve` 用于正文 Chunk 证据，由 Agent 组织答案。

写入型操作需要 confirm，并通过 JobManager 记录异步状态。读取型检索返回结构化证据，不替代上层答案生成。

## Agent 答案组织

MCP 作为 Agent 工具执行检索，最终答案由宿主 Agent 生成。`library_search` 和 `library_citation` 返回确定性 `data.presentation.answer_text`，Agent 原样输出；`library_retrieve` 只返回 Chunk 正文 Evidence、来源定位和一个最终 `score`。证据卡片使用 `source_id`、`paper_id`、`chunk_id`、`section_path` 和页码定位，Agent 根据 `evidence[*].text` 组织正文答案并添加引用；Reason 合并窗口才额外带 `source_chunk_ids`。

MCP 不调用答案生成模型，不保存答案上下文，也不提供答案校验工具。Agent 需要自行保证每个事实都能在返回的 Chunk 中找到依据；证据不足时明确说明，不补写检索结果没有的内容。

## 验证方式

    paper-rag catalog sync --json
    paper-rag catalog status --json
    paper-rag catalog chunks --sample 50 --seed 20261005 --json
    paper-rag index status --json
    paper-rag index rebuild --mode auto --json
    paper-rag retrieve "概括这篇论文的方法" --task auto --mode hybrid --json

当前自动测试结果以本地测试命令为准；最近的真实正文回归和人工复核保存在 `record.md`。回归问题统一使用中文输入，保留英文技术术语；正文测试覆盖检索状态、证据回退、来源编号和语义/词法降级。

## 当前边界

需要继续优化章节号和 Stage 查询的精确排序、显式 Appendix 内部小节排序、少量原始 MinerU block 的词边界异常，以及更系统的 Recall@K、Precision@K 和证据连续性评测。
