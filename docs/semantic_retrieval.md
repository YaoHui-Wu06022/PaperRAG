# 正文 Chunk、引用图与语义检索

paper_catalog_sync 是唯一的派生索引入口。它读取 MinerU content_list.json，把 Abstract、Content、Appendix 建立为正文 Chunk；Reference 只写入 SQLite references 和 citation_edges。

LlamaIndex 从 SQLite Chunk 构建稳定 TextNode，节点的正文内容使用 `retrieval_text`。构建向量时，DashScope Embedding 的输入严格为 `retrieval_text`，不拼接节点 metadata，向量随后写入 Milvus。表格的原始 HTML 保留在 `text` 中，检索和向量化只使用去除 HTML 标签后的结构化表格文本。library_retrieve 支持 lexical、semantic 和 hybrid，混合模式使用 RRF。

Hybrid 结果按 `chunk_id` 去重，同时保留 lexical rank、semantic rank、原始 semantic score 和 RRF score。过滤先作用于候选论文和正文区域，再执行结果截断。中文问题先由 Query Rewriter 提取完整 `core_terms`，再翻译后进入 FTS5 词法分支；Milvus 始终接收原始问题，以保留跨语言语义匹配能力。Query Rewriter 输出为空或不可用时回退到原始问题。

正文检索的元数据约束通过 filters 传入，例如：

    {"year_from": "2018", "category": "cs.CL", "state": "ingested"}

“2018 年以后有哪些文章”应由 Agent 选择 library_search；如果元数据自由文本是中文，library_search 会先使用腾讯云翻译；“2018 年以后哪些论文使用注意力机制”应选择 library_retrieve，由 filters 约束论文集合，再检索正文。

引用关系始终通过 `library_citation` 的 `references`、`citations` 或 `graph` mode 查询，不经过正文检索。

表格检索表示更新后先执行 Catalog 同步；只有在确认新的 `retrieval_text` 后，才执行 LlamaIndex/Milvus Embedding 重建。两者是独立的数据生命周期步骤。
