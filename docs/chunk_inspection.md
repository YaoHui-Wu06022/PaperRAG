# Chunk 检查与本地调试

Catalog 同步时会在每篇论文的 MinerU 目录生成 `chunks.json`：

```text
data/sources/arxiv/<paper_id>/mineru/chunks.json
```

SQLite 仍是检索权威来源，JSON 是与同一批 Chunk 同步生成的可读副本。文件使用 UTF-8，包含 `chunk_id`、`section_path`、`text`、`retrieval_text`、页码、来源 Block、资源引用和内容哈希。同步失败时，旧 JSON 不会被覆盖。

## 查看命令

```text
paper-rag catalog chunks --paper-id 2106.09685 --json
paper-rag catalog chunks --sample 50 --seed 20261005 --json
paper-rag catalog chunks --sample 50 --type table --json
paper-rag catalog chunks --sample 50 --type image --json
paper-rag catalog chunks --sample 50 --type equation --json
```

`--sample` 输出类型、区域、资源引用和重复路径统计，并返回完整抽样内容。指定 `--seed` 后可以重复得到同一批样本。

## 章节前缀

Chunk 保留 MinerU 的章节树。例如 LoRA 4.1 的检索文本现在是：

```text
content
4 OUR METHOD
4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES
<正文或公式>
```

`section_path` 和 `retrieval_text` 使用同一份去重后的路径。正文切分仍按 1200 字符和 150 字符重叠执行，LlamaIndex 不会重新切块。结构化块在正文阅读顺序中保持独立类型，同时在 `retrieval_text` 中加入相邻正文的 `[context_before]` 和 `[context_after]`，避免公式、表格或图片脱离语义上下文。

References 后面的 `A`、`D.1`、`H.4` 等字母编号章节会被识别为 Appendix，并继续保留附录内部层级。

## 表格、图片和公式

- 表格保留 caption、HTML `table_body`、footnote 和图片资源路径；HTML 不被展平成普通段落。
- 图片和图表保留 caption、footnote、页码和 `asset_refs`。没有 caption 时保留资源路径并返回 `media_text_missing` 提示，不生成虚假描述；相邻正文仍作为检索上下文保存。
- 公式保留 MinerU LaTeX 和 `text_format=latex`，不与相邻正文强行合并。
- `source_blocks` 保存原始页码、边界框、类型和格式，便于回到 MinerU 结果查看。

当前抽样中，表格主要是 HTML，公式是 LaTeX，带 caption 的图片可以直接被检索；无 caption 的图表仍需要客户端读取 `asset_refs` 查看原图。
