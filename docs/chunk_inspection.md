# Chunk 检查与本地调试

Catalog 同步时会在每篇论文的 MinerU 目录生成 `chunks.json`：

```text
data/sources/arxiv/<paper_id>/mineru/chunks.json
```

SQLite 仍是检索权威来源，JSON 是与同一批 Chunk 同步生成的可读副本。文件使用 UTF-8，包含 `chunk_id`、`section_path`、`text`、`retrieval_text`、`retrieval_text_hash`、页码、来源 Block、资源引用和内容哈希。`retrieval_text_hash` 是增量 Embedding 缓存的检索文本指纹。同步失败时，旧 JSON 不会被覆盖。

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

表格的 `text` 是展示和追溯用的源文本，保留 MinerU 的原始 HTML；表格的 `retrieval_text` 是 SQLite FTS 和 Embedding 的输入，会把显式 `<th>` 或 `<thead>` 表头转换为 `Columns: ...` 和 `列名: 值` 行。没有显式表头的表格保留单元格顺序，不默认把第一行当列名。HTML 标签、属性、脚本和样式不会进入检索文本，`rowspan`/`colspan` 也不会复制展开。

References 后面的 `A.`、`D.1.`、`H.4` 等字母编号章节会被识别为 Appendix；标题末尾句点只用于编号格式，不会增加层级，A/B/C 会保持同级，D.1/D.2 会挂在 D 下。

`Acknowledgements`、`Acknowledgments` 和对应单数形式会被视为独立顶层章节，不会继承前一个数字章节。例如它的路径是 `content → Acknowledgements`，而不是 `content → 9 Broader Impacts → Acknowledgements`。

如果致谢之后没有显式的 `Appendix` 标题，则在进入 `References` 之前遇到 `A`、`A1`、`A.1`、`A2.1` 等字母编号标题时，会从该标题切换到 Appendix。致谢后的 `Contributions` 仍然是 `content` 下的独立章节；例如 GPT-3 论文的路径为 `content → Acknowledgements`、`content → Contributions`，随后 `appendix → A Details...`。Swin Transformer 的 `A1/A2/A3` 也会从 `appendix` 根开始。

## 表格、图片和公式

- 表格 `text` 保留 caption、HTML `table_body`、footnote 和图片资源路径；HTML 不被展平成普通段落。`retrieval_text` 使用 caption、纯文本表头/数据行和 `Footnote:` 结构化表示。
- 图片和图表保留 caption、footnote、页码和 `asset_refs`。没有 caption 时保留资源路径并返回 `media_text_missing` 提示，不生成虚假描述；相邻正文仍作为检索上下文保存。
- 公式保留 MinerU LaTeX 和 `text_format=latex`，不与相邻正文强行合并。
- `source_blocks` 保存原始页码、边界框、类型和格式，便于回到 MinerU 结果查看。

当前抽样中，表格主要是 HTML，公式是 LaTeX，带 caption 的图片可以直接被检索；无 caption 的图表仍需要客户端读取 `asset_refs` 查看原图。
