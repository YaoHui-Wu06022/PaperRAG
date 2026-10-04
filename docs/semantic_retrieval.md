# 正文 Chunk、引用图与语义检索

`paper_catalog_sync --json` 是唯一的派生索引入口。它读取每篇 ArXiv 目录中的 MinerU `content_list.json`，跳过标题/作者/页眉页脚等元数据，按 `Abstract`、`Content`、`Appendix` 建立 Chunk；`Reference` 只写入引用表和引用边。

正文 Chunk 不跨章节合并。普通文本目标为 1200 个 Unicode 字符，章节内部才允许最多 150 个字符重叠；公式、表格、代码、列表和图片说明保持独立。MinerU 的 `page_idx` 以零基值写入 `source_blocks` 和 `page_start/page_end`，读取接口同时返回面向用户的加一页码字段。

向量索引使用 DashScope OpenAI 兼容接口生成 `qwen3.7-text-embedding-flash` 的 1024 维向量，写入可选的 Milvus。Embedding 和 Milvus 都是延迟加载的；未配置时词法 FTS5 仍可用。`paper_search_chunks` 的 `mode` 可选 `lexical`、`semantic` 或 `hybrid`，混合模式使用 RRF 合并排名。
