# MinerU 正文与引用图

Catalog 同步读取 MinerU 结果并按阅读顺序生成 Chunk。正文区域包括 abstract、content 和 appendix；reference 区域不生成正文检索证据，只解析为引用条目和引用边。

Chunk 保留稳定的 chunk_id、章节路径、页码、来源 Block、资源引用、原文和检索文本。LlamaIndex 不重新切块。

正文问题使用 library_retrieve。引用关系、被引用论文和发展关系使用 SQLite 图工具：

    library_get_references
    library_get_citations
    library_get_citation_graph

服务端只返回证据和结构化图数据，客户端负责最终回答。
