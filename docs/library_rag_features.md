# 本地论文库功能说明

## 1. 工具选择边界

Agent 先根据 MCP 工具描述选择顶层能力：

| 用户问题 | 工具 |
| --- | --- |
| 2018 年以后有哪些论文 | library_search |
| 2018 年以后哪些论文使用注意力机制 | library_retrieve，并传入 filters |
| 这个方法为什么有效 | library_retrieve，task=reason |
| 总结指定论文 | library_retrieve，task=summary |
| 比较两篇论文 | library_retrieve，task=comparison |
| 作者、版本、分类、资产状态 | library_get_metadata 或 library_get_asset_status |
| 参考文献、被引论文、发展关系 | library_get_references、library_get_citations、library_get_citation_graph |
| 阅读全文或指定 Chunk | library_read、library_get_chunk |

系统不注册 library_context 或顶层 library_route。library_retrieve 是唯一正文 RAG 工具。

## 2. 正文 RAG

请求示例：

    {
      "query": "哪些论文使用了注意力机制？",
      "filters": {"year_from": "2018", "state": "ingested"},
      "task": "auto",
      "mode": "hybrid",
      "limit": 8
    }

task 可选：

- auto：已经进入正文检索后调用 JEV；
- fact：返回事实证据；
- reason：返回直接证据，并补充同章节或相邻 Chunk；
- summary：每篇候选论文优先返回 abstract，再补充正文 Chunk；
- comparison：需要至少两篇唯一论文，按论文分别分配上下文预算。

JEV 只处理 RouteIntent.RETRIEVE 内部的正文任务，不选择 MCP 工具，不调用答案生成模型。JEV 返回的置信度会原样保留；调用失败时使用本地规则并在 warnings 中说明原因。

## 3. 元数据的两种作用

纯元数据问题直接使用 library_search。例如“2018 年以后有哪些文章”只查询 SQLite papers_fts 和结构化过滤条件，不读取正文。

正文问题可以把元数据作为检索约束。library_retrieve.filters 支持 author、category、year、year_from、year_to、state；系统先筛选论文，再在候选论文的正文 Chunk 中执行 lexical、semantic 或 hybrid 检索。paper_ids 支持 base ID 和 canonical ID，重复版本会合并；regions 只允许 abstract、content、appendix。

## 4. 引用图

Catalog 同步时从 MinerU Reference 区域解析 ArXiv ID、DOI 和原始引用文本，写入 references 与 citation_edges。这些数据不进入正文 Chunk 向量索引。

library_get_citation_graph 支持：

    {
      "paper_id": "1706.03762",
      "direction": "both",
      "depth": 2
    }

depth 最大为 3，图查询使用 SQLite BFS；查询引用关系不会触发 Embedding、Milvus 或正文检索。

## 5. 索引与运行

    paper-rag catalog sync --json
    paper-rag index rebuild --json
    paper-rag index status --json
    paper-rag search "attention" --json
    paper-rag retrieve "这个方法为什么有效" --task reason --mode hybrid --json

重建采用临时 Collection 和 Manifest 原子切换。失败时保留当前 active Collection。语义索引或 Embedding 不可用时，系统继续返回 SQLite lexical 证据，并标记 embedding_unavailable 或 milvus_unavailable。

## 6. 返回结果

library_retrieve 返回 status/data/warnings/read_only 包装。data.items 中每个来源都包含 source_id、chunk_id、paper_id、canonical_id、章节、页码、正文和资源引用。客户端使用 [S1] 等标记生成最终回答。
