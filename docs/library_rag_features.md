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
| 作者、版本、分类、资产状态 | library_get_metadata |
| 参考文献、被引论文、发展关系 | library_citation |
| 阅读全文或指定 Chunk | library_read、library_get_chunk |

系统不注册 `library_context` 或顶层 `library_route`。只有 Agent 选择 `library_retrieve` 后，服务内部才进入 `RouteIntent.RETRIEVE`，JEV 才会参与正文任务分类。`library_retrieve` 是唯一正文 RAG 工具。

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

四种任务都只返回证据，不生成答案：fact 返回直接证据；reason 补充同论文同章节或相邻 Chunk；summary 按论文覆盖摘要和正文；comparison 按论文均衡分配证据和上下文预算。

JEV 只处理 RouteIntent.RETRIEVE 内部的正文任务，不选择 MCP 工具，不调用答案生成模型。JEV 返回的置信度会原样保留；调用失败时使用本地规则并在 warnings 中说明原因。

## 3. 元数据的两种作用

纯元数据问题直接使用 library_search。例如“2018 年以后有哪些文章”只查询 SQLite papers_fts 和结构化过滤条件，不读取正文。自由文本查询先通过 OpenAI 兼容的 Query Rewriter 提取完整核心检索短语，再由腾讯云翻译后查询论文元数据。Query Rewriter 只返回 `core_terms` 数组，例如“注意力机制”返回 `{"core_terms": ["注意力机制"]}`，不拆分成独立的“注意力”和“机制”，也不修改 filters。Query Rewriter 不可用时回退到原始查询翻译和停用词清洗；作者、分类、年份和状态等结构化 filters 不参与翻译。

正文问题可以把元数据作为检索约束。library_retrieve.filters 支持 author、category、year、year_from、year_to、state；系统先筛选论文，再在候选论文的正文 Chunk 中执行 lexical、semantic 或 hybrid 检索。paper_ids 支持 base ID 和 canonical ID，重复版本会合并；regions 只允许 abstract、content、appendix。

## 4. 引用图

Catalog 同步时从 MinerU Reference 区域解析 ArXiv ID、DOI 和原始引用文本，写入 references 与 citation_edges。这些数据不进入正文 Chunk 向量索引。

引用目标采用本地混合匹配：先匹配规范化 ArXiv ID，再匹配规范化 DOI，最后使用标题、作者和年份进行严格相似匹配。`references` 保留所有原始引用，并记录 `matched_paper_id`、`match_method`、`match_score`、`resolution` 和 `duplicate_of_reference_id`；只有目标存在于本地 Catalog 且达到高置信度时，才写入 `citation_edges`。同一篇论文重复出现在原始参考文献中时，只保留一个主引用建立图边，其余条目保留并指向主引用。外部引用和歧义引用仍可查看，但不会污染本地图。

Catalog 同步结果中的 `citation_match_stats` 会分别统计 `arxiv_exact`、`doi_exact`、`title_author_year`、`external`、`ambiguous` 和 `unresolved`。引用匹配不调用外部服务，也不触发 Embedding、Milvus 或正文检索。

统一工具 `library_citation` 支持：

    {
      "paper_id": "1706.03762",
      "mode": "graph",
      "direction": "both",
      "depth": 2
    }

depth 最大为 3，图查询使用 SQLite BFS；查询引用关系不会触发 Embedding、Milvus 或正文检索。

每次 Catalog 同步还会生成 UTF-8 文件 `data/index/citation_graph.json`，包含本地论文节点、唯一引用边、引用数量和匹配统计，便于直接查看或导入图分析工具。原始重复参考条目仍在 SQLite `references` 中，并通过 `duplicate_of_reference_id` 指向主条目；JSON 图不会重复建边。

## 5. 索引与运行

    paper-rag catalog sync --json
    paper-rag index rebuild --json
    paper-rag index status --json
    paper-rag search "attention" --json
    paper-rag retrieve "why is attention useful" --task auto --mode hybrid --limit 8 --max-chars 12000 --json
    paper-rag retrieve "what evidence supports the method" --task fact --mode lexical --json
    paper-rag retrieve "compare the training methods" --task comparison --paper-id 2106.09685 --paper-id 2305.14314 --json

查看 JEV 和召回结果时关注：

```text
data.routing.route_intent
data.routing.task
data.routing.provider
data.routing.fallback_used
data.routing.confidence
warnings
data.items
data.context_text
```

`task=auto` 调用 JEV；显式指定四种任务时跳过 JEV。`items` 保留 lexical/semantic 排名、RRF 分数、章节、页码和资源引用。

Chunk 检查：

    paper-rag catalog chunks --paper-id 2106.09685 --json
    paper-rag catalog chunks --sample 50 --seed 20261005 --json

详见 `docs/chunk_inspection.md`。

重建采用临时 Collection 和 Manifest 原子切换。失败时保留当前 active Collection。语义索引或 Embedding 不可用时，系统继续返回 SQLite lexical 证据，并标记 embedding_unavailable 或 milvus_unavailable。

## 6. 返回结果

library_retrieve 返回 status/data/warnings/read_only 包装。data.items 中每个来源都包含 source_id、chunk_id、paper_id、canonical_id、章节、页码、正文和资源引用。客户端使用 [S1] 等标记生成最终回答；证据编号和页码组织方式参考 [PaperQA 示例输出](https://github.com/future-house/paper-qa#example-output)。

元数据和引用工具会在 `data.presentation` 中返回确定性答案模板。`library_search` 与 `library_citation` 的 `render_policy` 为 `verbatim`，Agent 应原样输出 `answer_text`，不重新总结或补充。参考文献答案只显示本地匹配数量和前 10 条本地论文题目；被引论文答案只显示数量和前 10 个来源论文题目；引用图答案显示方向、深度、节点数、边数以及前 10 条“引用方 引用 被引用方”关系。题目从本地 Catalog 读取，缺失时才回退到论文 ID。

`library_retrieve(mode="hybrid")` 的 `render_policy` 为 `compose`，Agent 可以根据正文证据、`source_id` 和 `context_text` 组织最终答案。lexical 和 semantic 模式仍只提供证据上下文。MCP 服务端不调用答案生成模型。
