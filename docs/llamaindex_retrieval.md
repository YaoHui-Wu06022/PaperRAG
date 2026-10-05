# LlamaIndex 正文 RAG

Agent 根据 MCP 工具描述选择能力。library_search 只做论文元数据查询；Agent 选择 library_retrieve 后，服务内部才进入 RouteIntent.RETRIEVE，并可由 JEV 将正文问题分类为 fact、reason、summary 或 comparison。

library_retrieve 是唯一正文 RAG 工具。它先应用 paper_ids 和 filters 元数据约束，再对 abstract、content、appendix 执行 SQLite lexical、Milvus semantic 或 hybrid RRF 检索；Milvus 语义查询使用相同的 paper_id/region metadata filters。reference 区域永远不进入正文向量索引或正文召回。

返回结果包含：

- task、papers 和 items；
- source_id、Chunk ID、论文 ID、章节、页码和资源引用；
- lexical_rank、semantic_rank、rrf_score；
- 面向客户端的 context_text 和 truncated。

表格、公式、图片和图表仍按正文顺序作为结构化 Chunk 保存；`retrieval_text` 额外包含相邻正文上下文，原始 `text`、HTML/LaTeX 和 `asset_refs` 保持独立。

任务策略如下：`fact` 返回直接证据；`reason` 增加相邻 Chunk；`summary` 按论文覆盖摘要和正文；`comparison` 按论文均衡分配来源和字符预算。`paper_ids` 与 `filters` 会在 lexical 和 semantic 两条路径使用相同的候选范围。

服务端不生成摘要、比较结论或问答答案。JEV 不选择 MCP 工具，也不进入 Embedding、Milvus 或 SQLite 底层检索。

首次使用：

    paper-rag catalog sync --json
    paper-rag index rebuild --json
    paper-rag index status --json
