# Chunk、Catalog 与检索索引

本文说明论文从 MinerU 解析结果到 Chunk、SQLite Catalog、LlamaIndex 和 Milvus 的完整数据链路，并给出如何检查切分质量的方法。

当前切分规则版本为 `content-list-regions-v6-boundary-aware`，主要参数为：

```text
MAX_CHARS = 1200
MAX_OVERLAP = 150
```

Chunk 的事实来源是 MinerU 的 `content_list.json`。SQLite `chunks` 表是检索时的权威副本；每篇论文目录下的 `chunks.json` 是与 SQLite 同批生成的人类可读副本，用于抽样、调试和人工检查。

## 1. 完整数据流

```text
PDF
  ↓ MinerU
content_list.json
  ↓ paper_rag/catalog/chunks.py
区域识别、章节树维护、正文切分、结构化块保留
  ├── SQLite chunks / chunks_fts
  ├── 每篇论文 mineru/chunks.json
  └── LlamaIndex TextNode
          ↓
      DashScope Embedding
          ↓
      Milvus active collection
```

引用区域不进入正文 Chunk。MinerU 的 Reference 内容由引用解析器提取到 SQLite `references` 和 `citation_edges`，引用图查询不会读取正文 Chunk、FTS、Embedding 或 Milvus。

Chunk 生成入口是 `paper_rag/catalog/chunks.py` 的 `build_chunks`。Catalog 同步时先在临时 SQLite 数据库中完成全部写入和校验，再原子替换正式数据库；Chunk JSON 也先写临时文件，成功后再替换正式文件。

## 2. Chunk 数据模型

每个 Chunk 至少包含以下字段：

| 字段 | 作用 |
| --- | --- |
| `chunk_id` | 稳定的 Chunk 主键，也是 LlamaIndex `TextNode.id_` 和 Milvus 中的逻辑 ID |
| `paper_id` | 当前论文的基础 ArXiv ID |
| `canonical_id` | 论文的版本 ID，例如带 `v1`、`v2` 的 ID |
| `ordinal` | 按 MinerU 原始顺序生成的序号 |
| `region` | `abstract`、`content` 或 `appendix`；`reference` 不生成正文 Chunk |
| `chapter_number` | 当前章节编号，例如 `4.1`、`A2.1` |
| `chapter_title` | 当前章节标题 |
| `section_path` | 从区域到当前标题的完整目录路径 |
| `section_label` | `section_path` 最后一个标题 |
| `type` | `text`、`table`、`equation`、`image`、`figure`、`chart`、`code`、`list` 等 |
| `text` | 展示和追溯用的原始内容 |
| `retrieval_text` | SQLite FTS5 和 Embedding 的输入文本 |
| `page_start`、`page_end` | 来源页码 |
| `source_blocks` | 对应的 MinerU block 索引、类型、页码及结构化字段 |
| `asset_refs` | 图片、表格或其他媒体资源路径 |
| `content_hash` | `text` 的 SHA-256，用于源内容追踪 |
| `retrieval_text_hash` | `retrieval_text` 的 SHA-256，用于 Embedding 缓存判断 |

`chunk_id` 使用论文版本、源内容哈希、规则版本、序号和原始 `text` 计算：

```text
sha256([canonical_id, content_hash, chunk_rule_version, ordinal, text])[:24]
```

因此，表格 HTML 转成纯文本检索表示时，`text` 不变则不会因为检索格式变化额外改变源内容身份；但 `retrieval_text_hash` 会变化，增量索引会据此重新计算 Embedding。

`paper_id` 不会加入 `retrieval_text` 前缀。论文身份通过 Node metadata、`paper_id` 和 Catalog 记录保存。

## 3. 区域识别与章节树

### 3.1 区域状态

解析状态按以下顺序变化：

```text
metadata → abstract → content → appendix
                         ↘ reference（终止正文 Chunk）
```

- `metadata`：论文标题、作者、机构和其他元数据，不生成正文 Chunk。
- `abstract`：识别显式 `Abstract` 或 `Abstract - ...`；部分 MinerU 结果没有 Abstract 标题时，会根据首个正文章节前的长文本推断摘要。
- `content`：正文章节和正文段落。
- `appendix`：显式 Appendix/Appendices，或满足致谢后隐式附录规则的章节。
- `reference`：References、Bibliography 等区域只交给引用解析器，不生成正文 Chunk。

### 3.2 正文章节

章节标题由 MinerU 的 `heading`、`section_header` 或带层级信息的文本块识别。数字章节支持：

```text
1 Introduction
4 OUR METHOD
4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES
```

`section_path` 会保留完整目录树，并清理相邻重复项。例如 LoRA 的正文 Chunk 会使用：

```text
content
4 OUR METHOD
4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES
```

不会加入论文 ID，也不会重复追加已经存在的区域或标题。

章节层级按照编号深度维护：`4` 是一级，`4.1` 是 `4` 的子级，`4.1.1` 是 `4.1` 的子级。无编号标题仍会保留在当前标题栈中，但不会强行伪造章节编号。

### 3.3 致谢与隐式 Appendix

`Acknowledgement`、`Acknowledgements`、`Acknowledgment` 和 `Acknowledgments` 会被当成独立的正文章节。致谢之后的 `Contributions` 等普通标题会重新从正文顶层开始，不会挂在上一章下面。

如果致谢之后、References 之前出现字母编号标题，系统可以把第一个符合条件的标题作为隐式 Appendix 起点。支持：

```text
A Details of ...
A. Details of ...
A1. Detailed Architectures
A2. Detailed Experimental Settings
A2.1. Image classification on ImageNet-1K
C.1 Additional Results
```

层级规则如下：

```text
A、B、C                 同级附录
A1、A2、A3              同级附录
A2.1、A2.2             A2 的子级
D.1、D.2               D 的子级
```

正文前面出现的 `A. Method` 或 `B. Setting` 不会自动识别为 Appendix；只有在致谢之后才启用这条隐式识别规则。References 仍然与正文和 Appendix 分离。

## 4. 普通正文切分

普通文本先按原始 block 顺序合并为逻辑段落，再执行切分：

1. 以空行区分段落单元；
2. 超过 1200 字符时，优先按中文句号、问号、感叹号以及英文句末边界切分；
3. 单句仍然过长时，在空白或词边界切分；
4. 没有空白的长 token 尽量保持完整，不从英文单词、数字或 LaTeX 命令中间截断；
5. 相邻 Chunk 复用完整句子或完整边界片段，重叠不超过 150 字符。

切分后的普通正文 Chunk 的 `text` 是原始正文片段，不带章节前缀。`retrieval_text` 才会加入章节路径，目的是让词法检索和语义检索知道 Chunk 所在的章节。

## 5. 结构化内容如何与正文保持顺序

表格、公式、图片、图表、代码和列表不会被拼接成一段连续的普通正文，而是各自保留为独立 Chunk。它们仍按 MinerU 原始位置参与 `ordinal` 排序，因此在论文阅读顺序中仍然穿插在正文之间：

```text
正文 Chunk（ordinal=10）
表格 Chunk（ordinal=11）
正文 Chunk（ordinal=12）
公式 Chunk（ordinal=13）
图片 Chunk（ordinal=14）
正文 Chunk（ordinal=15）
```

这种方式同时满足两个目标：原始结构不被破坏，表格/公式/图片又可以通过独立的类型、页码和资源引用精确追溯。

结构化 Chunk 会尝试提取附近的正文上下文：向前和向后最多检查 4 个 block，遇到标题或另一个结构化 block 就停止，最多分别保留前后 320 个字符。上下文只进入 `retrieval_text`，不会改写 `text`。

示例：

```text
content
3 Method
3.2 Training
[context_before]
We train the model with ...
<结构化内容的检索表示>
[context_after]
The ablation results are shown in Table 2.
```

因此，表格、公式和图片的检索语义可以结合前后正文，但不会因为上下文复制而改变原始媒体内容。

## 6. 表格 Chunk

### 6.1 展示内容与检索内容分离

- `text`：优先保留 MinerU 提供的独立 `text` 字段；没有时按 `table_caption`、`table_body`、`table_footnote` 拼接，原始 HTML 完整保留。
- `retrieval_text`：将 HTML 转为结构化纯文本，只供 FTS5 和 Embedding 使用。
- HTML 标签、属性、`script`、`style` 不进入检索文本。
- 不展开 `rowspan`/`colspan`，避免对复杂布局进行错误推断。

### 6.2 表头和数据行

只有显式 `<th>` 或 `<thead>` 才被当作表头，不会默认把第一行当表头。

输入示例：

```html
<table>
  <caption>Performance comparison</caption>
  <thead><tr><th>Model</th><th>Accuracy</th><th>Params</th></tr></thead>
  <tbody>
    <tr><td>BERT</td><td>89.2</td><td>110M</td></tr>
    <tr><td>RoBERTa</td><td>91.3</td><td>125M</td></tr>
  </tbody>
</table>
```

检索表示为：

```text
Performance comparison
Columns: Model | Accuracy | Params
Model: BERT | Accuracy: 89.2 | Params: 110M
Model: RoBERTa | Accuracy: 91.3 | Params: 125M
```

缺少单元格时补空位；多出的单元格使用稳定的 `Column N` 名称保留。例如：

```text
Columns: Model | Score
Model: A | Score: 0.8 | Column 3: extra value
```

脚注会以以下形式追加：

```text
Footnote: Results are averaged over three runs.
```

HTML 实体会解码，嵌套标签会去除，空白会压缩。HTML 不完整或无法解析行列时，回退为去标签后的纯文本，不让 Catalog 同步失败。

## 7. 图片、图表和公式 Chunk

### 图片和图表

图片/图表 Chunk 保留：

- `image_caption` 或 `chart_caption`；
- footnote；
- `img_path`、`image_path` 或其他 `asset_refs`；
- 页码、章节路径和原始 `source_blocks`；
- 附近正文的 `[context_before]` 和 `[context_after]`。

有 caption 时，caption 会进入 `text` 和 `retrieval_text`。没有 caption 时不生成虚假的视觉描述，保留媒体路径作为可追溯信息，并返回 `media_text_missing` warning。客户端可以通过 `asset_refs` 读取原始图片。

### 公式

公式保持 MinerU 提取的 LaTeX，不转换成普通段落，也不与相邻正文强行合并。公式 Chunk 的 `source_blocks` 中保留 `text_format=latex`、公式编号和页码等信息，章节前缀和相邻上下文提供公式的语义定位。

例如公式位于 `D.2. Approach 3` 时，`section_path` 应为：

```text
appendix
D. Details on the scaling analyses
D.2. Approach 3: Parametric fitting of the loss
```

而不是把 A、B、C 等其他附录标题全部叠加到同一条路径中。

## 8. `retrieval_text` 的确切组成

普通正文或结构化 Chunk 的检索文本由以下部分按顺序组成：

```text
<region>
<section_path 中的章节标题，每行一项>
[context_before]
<前文上下文>
<检索正文或表格纯文本>
[context_after]
<后文上下文>
```

没有上下文时不会产生空的标记。`paper_id` 不在这里出现；论文过滤由 SQLite Catalog 和 Retriever 的 metadata 条件完成。

因此：

- FTS5 搜索的是 `retrieval_text`；
- LlamaIndex `TextNode.text` 等于 `retrieval_text`；
- Milvus 向量化的是 `retrieval_text`；
- `text`、`source_blocks` 和 `asset_refs` 用于展示、引用和追溯。

## 9. SQLite、`chunks.json` 与索引生命周期

Catalog 同步时，SQLite 和每篇论文的 JSON 使用同一批 `Chunk` 对象生成，避免两套数据出现数量或字段不一致。

单篇 JSON 路径：

```text
data/sources/arxiv/<paper_id>/mineru/chunks.json
```

文件结构：

```json
{
  "schema_version": 1,
  "paper_id": "2106.09685",
  "canonical_id": "2106.09685v1",
  "content_hash": "...",
  "chunk_rule_version": "content-list-regions-v6-boundary-aware",
  "generated_at": "...",
  "chunk_count": 24,
  "chunks": []
}
```

Catalog 或 JSON 生成失败时，旧的正式文件保留不变。Catalog 同步本身不调用 Embedding，也不连接 Milvus。

只有 `index rebuild` 才会更新向量索引：

```text
paper-rag index rebuild --mode auto --json
paper-rag index rebuild --mode incremental --json
paper-rag index rebuild --mode full --json
```

Embedding 缓存至少按以下字段判断是否复用：

```text
chunk_id
retrieval_text_hash
embedding_model
embedding_dimensions
chunk_rule_version
```

表格检索格式、章节规则或其他切分规则变化时，`chunk_rule_version` 或 `retrieval_text_hash` 会触发对应 Chunk 更新；规则版本发生全局变化时，`auto` 会选择全量重建。改变引用图不会触发 Embedding 重建。

## 10. 查看和抽样检查

同步 Catalog：

```text
paper-rag catalog sync --json
```

查看单篇论文全部 Chunk：

```text
paper-rag catalog chunks --paper-id 2106.09685 --json
```

固定随机种子抽样：

```text
paper-rag catalog chunks --sample 50 --seed 20261005 --json
```

按类型抽样：

```text
paper-rag catalog chunks --sample 50 --type table --json
paper-rag catalog chunks --sample 50 --type image --json
paper-rag catalog chunks --sample 50 --type equation --json
```

抽样结果应重点检查：

1. `ordinal` 是否保持原文顺序；
2. `region` 是否正确区分正文、附录和引用；
3. `section_path` 是否从区域开始且没有重复尾部；
4. 长文本是否在句子或词边界切分；
5. overlap 是否保持语义连续且不超过 150 字符；
6. 表格 `text` 是否仍保留 HTML，`retrieval_text` 是否不含 HTML 标签；
7. 图片/图表是否保留 caption、资源路径和页码；
8. 公式是否保留 LaTeX；
9. `source_blocks` 和 `asset_refs` 是否可以回到 MinerU 原始文件；
10. SQLite `chunks` 数量是否与各论文 `chunks.json` 的 `chunk_count` 一致。

## 11. 常见现象与判断方式

### 表格 HTML 在 JSON 中存在，但检索文本没有 HTML

这是预期行为。HTML 属于展示和追溯内容；FTS5 和 Embedding 使用结构化纯文本，避免标签和布局属性污染检索。

### 图片出现 `media_text_missing`

表示 MinerU 没有提供 caption 或正文描述。系统仍保留媒体 Chunk 和 `asset_refs`，但不会凭空生成图片含义。客户端需要读取原始图片或依赖其他多模态流程。

### 引用很多，但引用图只有少量边

引用图只保存本地 Catalog 中成功匹配的论文。外部、未解析或有歧义的引用仍保存在 `references`，但不会伪装成本地图边。

### 章节路径包含完整目录树

这是为了让 `4.1` 下的 Chunk 在 lexical 和 semantic 检索中同时获得 `4 OUR METHOD` 与 `4.1` 的语义信息。路径只保留真实标题，系统会去除相邻重复项。

### 修改 Chunk 规则后为什么需要重新建向量

规则改变可能改变 `retrieval_text` 或 Chunk 边界。Catalog 同步只更新 SQLite 和 JSON；向量是否复用由 `retrieval_text_hash` 和 `chunk_rule_version` 判断，必要时再执行增量或全量 Embedding。

## 12. 相关实现位置

```text
paper_rag/catalog/chunks.py       # 区域识别、章节树、正文切分、表格纯文本化
paper_rag/catalog/service.py      # SQLite Catalog、chunks.json、原子替换
paper_rag/llamaindex/nodes.py     # Chunk → TextNode
paper_rag/llamaindex/index.py     # Milvus、Manifest、增量索引
paper_rag/llamaindex/retrievers.py # lexical、semantic、hybrid 检索
paper_rag/mcp/tools/catalog.py   # library_get_chunk、library_retrieve 等工具
```

Chunk 检查的原则是：原始内容可追溯、阅读顺序不改变、结构化内容不丢失、检索表示可搜索、引用关系与正文检索保持隔离。
