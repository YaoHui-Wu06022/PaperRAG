# MinerU 正文与引用图

Catalog 同步读取 MinerU 结果并按阅读顺序生成 Chunk。正文区域包括 abstract、content 和 appendix；Reference 区域不生成正文检索证据，只解析为引用条目和引用边。对于 References 后面的 `A`、`D.1`、`H.4` 等字母编号章节，解析器会切换回 appendix 区域并保留其层级。若论文在致谢后直接出现 `A`、`A1`、`A.1` 或 `A2.1` 等附录标题，且尚未进入 References，解析器也会从该标题切换到 appendix；致谢后的 `Contributions` 保持为 content 顶层章节。

Chunk 保留稳定的 chunk_id、章节路径、页码、来源 Block、资源引用、原文和检索文本。正文中的公式、表格、图片和图表按原阅读顺序保留为结构化 Chunk，并把相邻正文写入检索文本的上下文标记；LlamaIndex 不重新切块。表格 `text` 保留原始 HTML，`retrieval_text` 使用显式表头和数据行生成的结构化纯文本，HTML 不参与 FTS 或 Embedding。

Catalog 同步会在每篇论文的 `mineru/chunks.json` 生成可直接查看的副本。使用 `paper-rag catalog chunks --sample 50 --seed 20261005 --json` 可以抽样检查类型分布、章节路径和资源引用。

正文问题使用 library_retrieve。引用关系、被引用论文和发展关系使用 SQLite 图工具：

    library_citation(mode=references|citations|graph)

服务端只返回证据和结构化图数据，客户端负责最终回答。
