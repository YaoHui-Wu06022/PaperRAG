# Chunk、Catalog 与检索索引

本文说明论文从 MinerU 解析结果到 Chunk、SQLite Catalog、LlamaIndex 和 Milvus 的完整数据链路，并给出如何检查切分质量的方法。

当前切分规则版本为 `content-list-regions-v6-boundary-aware`，主要参数为：

```text
MAX_CHARS = 1200
MAX_OVERLAP = 150
```

Chunk 的事实来源是 MinerU 的 `content_list.json`。SQLite `chunks` 表是检索时的权威副本；每篇论文目录下的 `chunks.json` 使用相同输入和规则生成，是人类可读的检查副本。当前同步会分别调用切分函数生成 SQLite 和 JSON，并非直接复用同一组内存对象；同步期间不应同时修改解析源文件。

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
| `ordinal` | 从 0 开始，按 MinerU 原始顺序生成的序号 |
| `region` | `abstract`、`content` 或 `appendix`；`reference` 不生成正文 Chunk |
| `chapter_number` | 当前章节编号，例如 `4.1`、`A2.1` |
| `chapter_title` | 当前章节标题 |
| `section_path` | 从区域到当前标题的完整目录路径 |
| `section_label` | `section_path` 最后一个标题 |
| `type` | `text`、`table`、`equation`、`image`、`figure`、`chart`、`code`、`list` 等 |
| `text` | 展示和追溯用的原始内容 |
| `retrieval_text` | SQLite FTS5 和 Embedding 的输入文本 |
| `page_start`、`page_end` | MinerU 零基来源页码；展示页码需加 1，未知可为 null |
| `source_blocks` | 对应的 MinerU block index、type、page_idx、bbox、text_format，以及媒体邻域上下文；不是全部原始字段的复制 |
| `asset_refs` | 图片、表格或其他媒体资源路径 |
| `content_hash` | `text` 的 SHA-256，用于源内容追踪 |
| `retrieval_text_hash` | `retrieval_text` 的 SHA-256，用于 Embedding 缓存判断 |

`chunk_id` 使用论文版本、源内容哈希、规则版本、序号和原始 `text` 计算：

```text
sha256(JSON([canonical_id, content_list_file_hash, chunk_rule_version, ordinal, text]))[:24]
```

这里的 content_list_file_hash 是整份解析文件的哈希，区别于表中 content_hash（单个 Chunk.text 的哈希）。身份计算不直接使用 retrieval_text：输入文件、规则、序号和 text 都不变时，仅检索表示变化不会改变 ID，但 retrieval_text_hash 会变化。同一篇重新解析后，整份文件哈希变化可能让所有 Chunk ID 改变；升级规则版本也会改变 ID，不能承诺跨解析版本的逐段内容去重。

chunk_rule_version 保存在 chunks.json 顶层、Catalog 元信息、Embedding 缓存和 Node metadata 中，当前不是 SQLite chunks 的独立列或每条 JSON Chunk 的字段。

`paper_id` 不会加入 `retrieval_text` 前缀。论文身份通过 Node metadata、`paper_id` 和 Catalog 记录保存。

## 3. 区域识别与章节树

### 3.1 区域状态

解析状态按以下顺序变化：

```text
metadata → abstract → content → appendix
                         ↘ reference（暂停正文 Chunk）
                              ↘ appendix（后续附录标题）
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

正文前面出现的 `A. Method` 或 `B. Setting` 不会自动识别为 Appendix；致谢之后才启用该隐式规则。另外，Reference 区域之后遇到显式 Appendix 或有效字母编号标题，可以恢复到 appendix，覆盖参考文献之后才开始附录的排版。字母编号和标题正文之间需要空格，C XXX 合法，CXXX 不按该规则识别。References 本身不生成正文 Chunk。

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

结构化 Chunk 会尝试提取附近的正文上下文：向前和向后最多检查 4 个 block，遇到标题停止；跳过页眉页脚及其他结构化块，找到最近的有效普通正文。最多分别保留前文末尾和后文开头 320 个字符。这里仍按字符截取，极长上下文可能出现半词；普通正文的边界切分并不代表媒体邻域也完全按词边界截取。上下文只进入 retrieval_text，不会改写 text。

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

输入示例：MinerU 的 table_caption 为 Performance comparison，table_body 为以下 HTML。caption 由独立字段加入检索表示：

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

当前行列解析器只收集单元格，没有额外提取纯 HTML 的 caption 节点。因此 caption 若只出现在 table_body 的 caption 标签中、没有 table_caption 字段，解析到有效行列时不会单独加入该标题；原 HTML 仍保留在 text。多行显式表头只用第一行生成 Columns，其余表头行不作为数据行输出；不推断多层列名关系。与论文标题完全相同的独立 caption 会过滤，减少误识别标题噪声。

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

图片/图表的展示文本、来源定位与检索表示分别保留：

- caption 和 footnote 组成的原始描述文本；
- `img_path`、`image_path` 等识别出的 `asset_refs`；
- 页码、章节路径和原始 `source_blocks`；
- 附近正文的 `[context_before]` 和 `[context_after]`。

有 caption 时，caption 会进入 text 和 retrieval_text。没有可用描述但有资源路径时，保留媒体 Chunk、路径及 media_text_missing warning；文本和路径都没有时会跳过该空媒体块并记录警告，不伪造图片语义。caption/footnote 并非都以原字段名完整复制到 source_blocks，原始结构应查看 content_list.json。客户端可用资源路径另行读取图片，正文 evidence 本身不包含图片像素。

### 公式

公式保持 MinerU 提取的 LaTeX，不转换成普通段落，也不与相邻正文强行合并。source_blocks 复制输入中已有的 text_format 和页码，不自动添加缺失格式。公式编号若在原始 LaTeX（如 tag）中则保留，不额外推断独立编号字段。章节前缀和相邻上下文提供公式的语义定位。

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

- chunks_fts 索引 retrieval_text，同时索引 section_path、region、chapter_title；不是单独搜索展示用 HTML text；
- LlamaIndex `TextNode.text` 等于 `retrieval_text`；
- Milvus 向量化的是 `retrieval_text`；
- `text`、`source_blocks` 和 `asset_refs` 用于展示、引用和追溯。

## 9. SQLite、`chunks.json` 与索引生命周期

Catalog 同步时，先扫描本地论文、在临时 SQLite 建库并校验；再读取相同解析输入生成各篇 JSON 临时文件和引用图 JSON。输入保持不变时，同一规则应产生相同 Chunk ID、数量和内容。同步期间改变源文件可能使两次读取不一致，不能把它视为一个持有原始文件快照的数据库事务。

单篇 JSON 路径：

```text
data/sources/arxiv/<paper_id>/mineru/chunks.json
```

文件结构：

下面是字段示意，chunks 数组内容省略；实际数组长度应等于 chunk_count。顶层 content_hash 对应解析源文件，单条 Chunk.content_hash 对应展示文本，二者不要混用。

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

准备阶段失败时旧正式文件保持不变。发布顺序为正式 SQLite → 各篇 chunks.json → citation_graph.json，每个文件独立原子替换；若后半段替换失败，前面已发布的文件不会整体回滚。因此 SQLite 与所有 JSON 不是一个多文件原子事务，需重新同步并核对。Catalog 同步本身不调用 Embedding，也不连接 Milvus。

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

检索文本变化由 retrieval_text_hash 判断，规则版本全局变化时 auto 选择全量。引用图查询或匹配规则不改变正文 Embedding；但重新同步 Catalog 会刷新 indexed_at，可能使索引状态暂时 stale。全量使用临时 Collection，增量当前原地更新 active，具体失败恢复边界见 [index.md](index.md)。

### Catalog 表与数据校验

| 表 | 用途 |
|---|---|
| papers / papers_fts | 元数据、资产路径和标题/摘要/作者/分类等发现；FTS 也索引 base/canonical ID |
| chunks / chunks_fts | 完整正文来源与全文检索；论文/区域过滤在 SQL LIMIT 之前生效 |
| references / references_fts | 原始引用条目、匹配状态和保留的外部引用 |
| citation_edges | 去重后的本地有向引用边 |
| embedding_items / embedding_state | 文档 Embedding 缓存身份与同步状态，不保存向量本体 |
| catalog_meta | indexed_at、规则版本、统计等派生状态 |

论文 metadata.json 扫描发生在重建阶段；元数据查询使用 SQLite，但构造记录时会检查该论文资产文件是否存在。state 来自同步快照，不等同于严格解析有效性。

MinerU 结果需存在 manifest.json、full.md、content_list.json，manifest 的论文版本、源 PDF 哈希、模型和语言要匹配。旧版本或不完整结果不会直接作为有效正文 Chunk 入库，相应问题见同步 issues。数据库构建会执行完整性检查，但 JSON 数量和字段仍可按上述命令人工抽查。

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

指定 paper-id 时读取该论文 chunks.json；全库抽样收集已有 JSON，sample 超过可用数量时返回全部，seed 使选择可复现。type 过滤可与单篇或随机抽样结合，不触发重新切块。统计描述本次选择的 Chunk：类型/区域分布、章节路径尾部相邻重复，以及有资源引用的 Chunk 数；资源引用数量不是图片总文件数。

JSON 缺失不会自动生成，先同步 Catalog。JSON 是检查副本，不应手工修改后期待 SQLite 或向量随之改变。

### 最终 evidence 为什么比 chunks.json 字段少

library_retrieve 返回紧凑证据：source_id、paper_id、chunk_id、text、type、section_path、page_start/page_end、score；reason 窗口另含 source_chunk_ids。章节前缀和媒体邻域用于召回，不直接复制为普通展示 text。公式的解释上下文可经 reason 窗口或其他正文来源获得，不能把只有公式的 Chunk 当作完整机制说明。

完整 retrieval_text、asset_refs、source_blocks、canonical_id、ordinal 等仍可通过 library_get_chunk 或 CLI read chunk 查看。原始页码为零基，完整 Chunk 读取另外提供 page_start_display/page_end_display；紧凑 evidence 当前没有展示页码字段。详情见 [retrieval.md](retrieval.md)。

## 11. 常见现象与判断方式

### 表格 HTML 在 JSON 中存在，但检索文本没有 HTML

这是预期行为。HTML 属于展示和追溯内容；FTS5 和 Embedding 使用结构化纯文本，避免标签和布局属性污染检索。

### 图片出现 `media_text_missing`

表示 MinerU 没有提供可用 caption 或正文描述。有路径时保留媒体 Chunk 和 asset_refs，无文本也无路径时跳过空块；不会凭空生成图片含义。客户端需要读取原始图片或依赖其他多模态流程。

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
paper_rag/mcp/tools/catalog.py   # library_get_chunk、引用和索引管理工具
paper_rag/mcp/tools/query.py     # library_search、library_retrieve
```

Chunk 检查的原则是：原始内容可追溯、阅读顺序不改变、结构化内容不丢失、检索表示可搜索、引用关系与正文检索保持隔离。
