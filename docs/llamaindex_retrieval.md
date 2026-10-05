# LlamaIndex 正文 RAG

Agent 根据 MCP 工具描述选择能力。library_search 只做论文元数据查询；自由文本先由 OpenAI 兼容 Query Rewriter 提取完整 `core_terms`，再在进入 SQLite FTS5 前使用腾讯云翻译，并在 query_debug 中返回改写和翻译状态。Agent 选择 library_retrieve 后，服务内部才进入 RouteIntent.RETRIEVE，并可由 JEV 将正文问题分类为 fact、reason、summary 或 comparison。

library_retrieve 是唯一正文 RAG 工具。它先应用 paper_ids 和 filters 元数据约束，再对 abstract、content、appendix 执行 SQLite lexical、Milvus semantic 或 hybrid RRF 检索；Milvus 语义查询使用相同的 paper_id/region metadata filters。reference 区域永远不进入正文向量索引或正文召回。

返回结果包含：

- task、papers 和 items；
- source_id、Chunk ID、论文 ID、章节、页码和资源引用；
- lexical_rank、semantic_rank、rrf_score；
- retrieval_debug：Query Rewriter 的 `core_terms`、改写回退状态、词法查询、翻译服务和停用词；
- 面向客户端的 context_text 和 truncated。

词法和语义检索使用不同的查询输入：原始问题直接进入 Milvus Embedding；词法分支先由 Query Rewriter 输出完整 `core_terms`，含中文时再通过腾讯云逐项翻译为英文。以“注意力机制”为例，实际 BM25 查询为精确短语 `"attention mechanism"`，不会拆成 `attention OR mechanism`。Query Rewriter 或腾讯云失败时词法分支回退到原始问题，语义检索仍照常执行。FTS5 查询阶段会过滤保守的英文停用词，并保留 BM25、RAG、模型名、数字和缩写等技术词。

表格、公式、图片和图表仍按正文顺序作为结构化 Chunk 保存；`retrieval_text` 额外包含相邻正文上下文，也是 Milvus Embedding 的唯一输入，节点 metadata 不参与向量化。表格的原始 HTML 只保存在 `text` 中用于展示和追溯；表格 `retrieval_text` 使用显式 `<th>`/`<thead>` 表头生成纯文本列名和数据行，不展开 `rowspan`/`colspan`。原始 `text`、HTML/LaTeX 和 `asset_refs` 保持独立。

任务策略如下：`fact` 返回直接证据；`reason` 增加相邻 Chunk；`summary` 按论文覆盖摘要和正文；`comparison` 按论文均衡分配来源和字符预算。`paper_ids` 与 `filters` 会在 lexical 和 semantic 两条路径使用相同的候选范围。

服务端不生成摘要、比较结论或问答答案。JEV 不选择 MCP 工具，也不进入 Embedding、Milvus 或 SQLite 底层检索。

首次使用：

    paper-rag catalog sync --json
    paper-rag index status --json

Catalog 同步会更新 SQLite 和每篇论文的 `chunks.json`。当确认新的 `retrieval_text` 需要进入向量库后，再执行：

    paper-rag index rebuild --mode auto --json

Catalog 更新本身不会自动重建 Embedding。索引同步使用 `embedding_state` 和 `embedding_items` 保存缓存键：`chunk_id`、`retrieval_text_hash`、`embedding_model`、`embedding_dimensions` 和 `chunk_rule_version`。`auto` 在兼容时只复制未变化 Chunk 的旧向量；新增或检索文本变化的 Chunk 才调用 DashScope。

## 增量索引

`index rebuild` 支持三种模式：

- `auto`：默认模式。缺少 Manifest、Milvus Collection、缓存不完整，或模型、维度、Chunk 规则变化时全量重建；其他情况增量同步。
- `incremental`：要求当前 active 索引和缓存兼容。不兼容时返回 `rebuild_required`，不会偷偷执行全量重建。
- `full`：强制为全部当前正文 Chunk 重新生成 Embedding。

MCP 客户端使用同一个 `library_index_rebuild(confirm=true, mode="auto")` 参数；它仍通过 JobManager 异步执行。

每次同步都会返回 `reused`、`added`、`updated`、`deleted`、`failed` 统计。同步先写入临时 Collection，验证数量、Chunk ID、向量维度和探测查询后才原子替换 Manifest；失败时保留原 active Collection。删除 Chunk 不会写入新 Collection，旧 Collection 在切换成功后清理。

可以用状态命令查看具体 stale 原因：

    paper-rag index status --json

状态中的 `stale_reasons` 会指出 `catalog_timestamp_mismatch`、`chunk_rule_version_mismatch`、`cache_incomplete` 等原因，而不是只返回一个布尔值。第一次使用当前表格纯文本规则时通常会看到全量统计；再次执行 `--mode auto` 应看到 `reused` 等于当前 Chunk 总数，其余四项为 0。
