# 正文 Chunk、引用图与语义检索

paper_catalog_sync 是唯一的派生索引入口。它读取 MinerU content_list.json，把 Abstract、Content、Appendix 建立为正文 Chunk；Reference 只写入 SQLite references 和 citation_edges。

LlamaIndex 从 SQLite Chunk 构建稳定 TextNode，使用 DashScope Embedding 写入 Milvus。library_retrieve 支持 lexical、semantic 和 hybrid，混合模式使用 RRF。

Hybrid 结果按 `chunk_id` 去重，同时保留 lexical rank、semantic rank、原始 semantic score 和 RRF score。过滤先作用于候选论文和正文区域，再执行结果截断。

正文检索的元数据约束通过 filters 传入，例如：

    {"year_from": "2018", "category": "cs.CL", "state": "ingested"}

“2018 年以后有哪些文章”应由 Agent 选择 library_search；“2018 年以后哪些论文使用注意力机制”应选择 library_retrieve，由 filters 约束论文集合，再检索正文。

引用关系始终通过 library_get_references、library_get_citations 或 library_get_citation_graph 查询，不经过正文检索。
