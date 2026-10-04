# MinerU 正文与 Chunk

`paper_arxiv_ingest` 只负责把云端 MinerU 的结果安全写入论文目录。`paper_catalog_sync(confirm=true)` 才读取 `mineru/content_list.json`，校验 manifest 与当前 PDF 版本后生成 SQLite Chunk 和 FTS5 索引。

Chunk 构建按阅读顺序保留四类区域：

- `abstract`：从 Abstract 标题开始，到正文第一节之前；
- `content`：正文章节；
- `reference`：References、Bibliography 等参考文献区域；
- `appendix`：Appendix/Appendices 区域。

ArXiv metadata 已包含标题、作者、摘要、分类和版本，因此 MinerU 解析中的前置标题、作者、页眉、页脚、页码和页脚注不进入正文召回。图片只保存说明文本和资源引用；表格、公式、代码和列表作为独立 Chunk。

正文读取接口：

```text
paper_search_chunks(query, paper_ids=None, limit=8)
paper_get_chunk(chunk_id)
paper_get_fulltext(paper_id, offset=0, limit=12000)
```

`paper_query` 的总结和比较意图返回按 Chunk 顺序组织的有来源上下文；正文意图使用 Chunk FTS5 召回。服务端不生成答案，最终回答由 Agent 根据返回的原文和来源组织。
