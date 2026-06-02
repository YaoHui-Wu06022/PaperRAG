# Paper_RAG

## 项目目标

核心目标是把本地 PDF 论文库整理成可重建、可检索、可追溯的结构化知识库，并在此基础上提供面向论文问题的 RAG 能力

设计原则：

- 原始 PDF、MinerU 原始输出、项目内部结构化数据分层保存
- 入库阶段沉淀稳定索引：`manifest`、`metadata`、`toc`、`blocks`、`chunks`、`references`、`citation_graph`、`annotations`
- 检索阶段先做语义路由和论文范围解析，再进入 `metadata`、`reference`、`content` 三条执行链路
- 对外 evidence 只保留回答组织需要的信息，内部路径、hash、raw records、raw chunks 等调试信息只在 `--debug` 下输出
- metadata/reference 尽量使用本地确定性回答，content 使用检索证据交给回答 LLM 生成最终答案

## 仓库结构与模块说明

```text
RAG_project/
├─ pyproject.toml                    # Python 包配置，声明 paper-rag CLI 入口
├─ README.md                         # 当前技术文档
├─ PLAN.md                           # 阶段性设计计划与模块规范
├─ .env.example                      # 环境变量模板
├─ data/                             # 本地论文数据与派生索引
├─ tests/                            # unittest 测试
└─ paper_rag/                        # 核心 Python 包
```

### 本地数据目录

```text
data/
├─ pdf/                              # 输入 PDF
├─ manifest.jsonl                    # 本地论文清单与状态
├─ index/                            # 检索派生索引与 embedding 缓存
│  ├─ embedding_cache.jsonl          # chunk embedding 缓存
│  ├─ query_embedding_cache.jsonl    # query embedding 缓存
│  └─ bm25_chunks.json               # BM25 corpus 派生索引
├─ mineru_output/                    # MinerU 原始解析结果
├─ paper_data/                       # 项目内部结构化论文数据
│  ├─ <paper_id>/
│  │  ├─ metadata.json               # 单篇论文元数据
│  │  ├─ toc.json                    # 目录树
│  │  ├─ blocks.jsonl                # 清洗后的结构化 block
│  │  ├─ chunks.jsonl                # 面向检索的 chunk
│  │  └─ references.jsonl            # 参考文献原始证据
│  └─ citation_graph.json            # 本地库内引用图
├─ paper_annotations.json            # 人工维护的论文 aliases/tags
└─ venue_aliases.json                # venue canonical/display/aliases 规则
```

### 基础入口与配置

这一组只负责包入口、配置读取和跨子系统通用工具，不承载具体业务流程

```text
paper_rag/
├─ __init__.py                       # 包标记
├─ __main__.py                       # `python -m paper_rag` 入口
├─ config.py                         # Settings 与 .env 读取、路径解析、默认配置
└─ utils.py                          # 根包通用工具：hash、slug、文本规范化、安全目录替换
```

### CLI 命令层

`cli/` 把命令行参数转换成内部服务调用，尽量保持薄封装，业务逻辑放回 `ingest/`、`retrieval/` 和 `answer/`

```text
paper_rag/cli/
├─ __init__.py                       # CLI 子包标记
├─ main.py                           # CLI 总入口与全局参数
├─ ingest.py                         # `paper-rag ingest`
├─ retrieval.py                      # `paper-rag index/search/plan`
└─ ask.py                            # `paper-rag ask`
```

### 入库处理

`ingest/` 负责把 PDF 和 MinerU 输出变成项目内部稳定数据结构

```text
paper_rag/ingest/
├─ __init__.py                       # 入库子包标记
├─ pipeline.py                       # 全量 PDF 同步、metadata 补全、extract、citation graph 主流程
├─ manifest.py                       # ManifestRecord/Manifest、状态管理、年份规范化
├─ mineru.py                         # MinerU API 上传、轮询、下载、解压
├─ extract.py                        # 从 MinerU 输出构建 metadata/toc/blocks/references/chunks
├─ citation_graph.py                 # 从 references 构建本地库内 citation graph
├─ annotations.py                    # paper_annotations.json 生成、规范化与保存
└─ metadata_sources/
   ├─ __init__.py                    # 元数据检索子包标记
   ├─ arxiv.py                       # ArXiv 精确标题查询与 preprint metadata
   ├─ dblp.py                        # DBLP 精确标题查询与正式发表 metadata
   ├─ semantic_scholar.py            # Semantic Scholar 正式发表信息补充
   └─ retry.py                       # 外部元数据请求的重试、退避、延迟
```

### 本地论文库访问层

`corpus/` 是检索和回答读取本地结构化数据的统一入口，负责本地数据文件加载成可查询的记录和范围

```text
paper_rag/corpus/
├─ __init__.py                       # 本地结构化论文库访问层
├─ aliases.py                        # 论文 mention/alias 到 canonical paper 的匹配
├─ annotation_index.py               # paper_annotations.json 的统一扫描入口
├─ chunks.py                         # chunks.jsonl 加载、ChunkDocument、按论文过滤
├─ citation_index.py                 # paper follow/prior 基于 citation graph 的范围解析
├─ context.py                        # 单次 plan/ask 内复用 manifest、chunks、BM25、citation graph
├─ filters.py                        # 单条 manifest record 的 filter 布尔匹配
├─ records.py                        # active manifest 读取、论文匹配、record key、去重
├─ scope.py                          # semantic + filters + groups 到候选论文 records
├─ resolver.py                       # parser 输出中的 paper/year/venue scope 标准化
├─ venues.py                         # venue canonical/display/aliases 规范化与匹配
└─ utils.py                          # token 规范化、去重、interval boundary、value 展平
```

### 检索与路由

`retrieval/` 负责把自然语言问题拆成可执行计划，并产出回答层消费的 evidence

包含三块：顶层编排、dense/BM25 召回、metadata/reference/content 三条语义路由

```text
paper_rag/retrieval/
├─ __init__.py                       # 检索子包标记
├─ plan.py                           # 编排：top route -> domain router -> planner
├─ route.py                          # RouteDecision，保存 parser 归一化后的路由状态
├─ evidence.py                       # composer/debug evidence 统一构建
├─ evidence_probe.py                 # evidence 调试入口
├─ timing.py                         # debug 模式下的分阶段耗时记录
└─ chunk_fusion.py                   # dense/BM25 命中结果的 RRF 融合
```

```text
paper_rag/retrieval/dense/
├─ __init__.py                       # dense 子包标记
├─ service.py                        # index/search/content dense search 高层服务；index 时写 Milvus 和 BM25 派生索引
├─ embedding.py                      # OpenAI-compatible embedding HTTP 客户端
├─ cache.py                          # embedding 本地缓存
└─ milvus_store.py                   # Milvus/Zilliz collection 重建、插入、向量搜索

paper_rag/retrieval/sparse/
├─ __init__.py                       # sparse 子包标记
└─ bm25.py                           # BM25CorpusIndex、派生索引读写、英文 token 规范化、多 query RRF 合并
```

`paper-rag index`由 `retrieval/dense/service.py` 编排：

- 读取 `paper_data/*/chunks.jsonl`
- 生成或复用 chunk embedding cache
- 重建 Milvus/Zilliz dense collection
- 同步写出 `data/index/bm25_chunks.json`

```text
paper_rag/retrieval/routes/
├─ __init__.py                       # route 子包标记
├─ common/                           # route 共用 parser client、prompt 片段、schema/filter/group 校验
├─ top/                              # 顶层路由，只分类 metadata/reference/content/unclear
├─ metadata/                         # 元数据问题：parser/router/planner/prompt probe
├─ reference/                        # 引用关系问题：source/object scope 修正与 citation graph 查询
└─ content/                          # 内容问题：检索 query 构建、dense/BM25/fusion、上下文扩展
```

### 答案生成

`answer/` 消费 retrieval evidence

metadata/reference 优先走本地确定性回答，content route 使用 LLM 基于证据生成自然语言答案

```text
paper_rag/answer/
├─ __init__.py                       # answer 包入口，re-export run_ask
├─ service.py                        # ask 薄编排：plan evidence -> local/LLM answer
├─ local.py                          # metadata/reference 的本地确定性回答
└─ llm.py                            # content route 的回答 LLM 客户端与 prompt 组装
```

## 配置

复制 `.env.example` 为 `.env`

主要分为几类配置

- `MinerU`
- `metadata`查询
- `Plan parser`
- `Answer composer`
- `Content retrieval`
- `Dense index`
- `BM25 keyword translation`

`ANSWER_*` 默认可复用 `PLAN_PARSER_*`

Embedding 缓存分两类：

- `EMBEDDING_CACHE_PATH`：chunk embedding 缓存，主要由 `index` 使用
- `QUERY_EMBEDDING_CACHE_PATH`：query embedding 缓存，主要由 `search/plan/ask` 使用

BM25 派生索引固定写入 `data/index/bm25_chunks.json`

它由 `index` 命令生成，可以删除后重建，不属于 `paper_data` 数据 schema

## CLI

`paper-rag` 和 `python -m paper_rag` 走同一套入口

命令形式：

```text
paper-rag [--project-root PROJECT_ROOT] <command>
```

- `--project-root PROJECT_ROOT`：指定项目根目录，用来定位 `.env`、`data/` 和索引配置

子命令：

```text
paper-rag ingest [--refresh]
paper-rag index
paper-rag search [--top-k TOP_K] <query>
paper-rag plan [--debug] <query>
paper-rag ask [--debug] [--json] <query>
```

### ingest

同步 `data/pdf/` 到本地结构化论文库

主要职责：

- 维护 `manifest.jsonl`
- 调用或复用 MinerU 输出
- 生成 `metadata.json`、`toc.json`、`blocks.jsonl`、`chunks.jsonl`、`references.jsonl`
- 更新 `citation_graph.json` 和 `paper_annotations.json`

参数：

- `--refresh`：重新刷新 active PDF 的外部 metadata

### index

根据 `data/paper_data/*/chunks.jsonl` 重建 dense 向量索引和 BM25 派生索引

只处理向量库，不重新解析 PDF，不刷新 metadata

主要职责：

- 新增或刷新论文后同步 dense index
- 生成 BM25 派生索引
- 修改 embedding 或 Milvus/Zilliz collection 配置后重建索引

### search

直接对 dense 向量索引做 chunk 搜索

参数：

- `<query>`：检索文本
- `--top-k TOP_K`：返回 chunk 数量，默认 `5`

输出包含 score、论文标题、section path、页码、chunk id 和 snippet

不经过 planner，不做 metadata/reference/content 路由

### plan

把用户问题解析成检索计划，并输出 answer 层消费的 evidence JSON

执行过程：

```text
query
  -> top route parser
  -> metadata/reference/content router
  -> route planner
  -> evidence
```

参数：

- `<query>`：要规划的问题
- `--debug`：输出中间状态和 `timings_ms`

### ask

面向最终使用的问答入口

回答策略：

- `metadata`：本地确定性回答
- `reference`：本地 citation graph 回答
- `content`：基于 evidence 调用回答 LLM
- `unclear` 或无证据：返回降级说明

参数：

- `<query>`：要回答的问题
- `--debug`：在 answer payload 中保留 planner debug 信息和 `timings_ms`
- `--json`：输出完整 JSON payload，不传时只打印 `answer`

常用命令示例：

```powershell
python -m paper_rag ingest
python -m paper_rag ingest --refresh
python -m paper_rag index
python -m paper_rag search "residual connection" --top-k 5
python -m paper_rag plan "ResNet 的模型结构是什么？" --debug
python -m paper_rag ask "发表在 CVPR 上的论文有哪些？"
python -m paper_rag ask "哪些论文引用了 ResNet？" --json
```

## 命名规范

### 函数命名

- `validate_*`：schema / parser payload 校验，失败抛 `PlanParseError`
- `normalize_*`：规范化内容，不读本地数据，不做检索
- `resolve_*`：把 parser mention、别名、venue、年份边界解析成内部稳定值
- `match_*`：布尔匹配
- `filter_*`：集合过滤并返回子集
- `search_*`：执行检索
- `build_*`：组装结构化对象或 evidence
- `to_evidence_*`：内部对象裁剪成对外 evidence 字段
- `dedupe_*`：按明确 key 保序去重

### 变量命名

- `query`：用户原始问题
- `retrieval_query`：content route 内部检索输入对象
- `dense_query`：embedding 使用的中文自然语言语义句
- `bm25_queries`：BM25 使用的关键词候选列表
- `route`：顶层语义路由，取值为 `metadata`、`reference`、`content`、`unclear`
- `parser_result`：LLM parser 的结构化输出
- `RouteDecision`：parser 结果经本地规范化后的不可变决策对象
- `paper_semantic / filters / paper_groups / group_mode`：单侧论文范围结构
- `source_*` / `object_*`：reference route 的两侧 scope，规范为 `source --cites--> object`
- `context_units`：content 检索命中 chunk 后扩展出的上下文单元
- `evidence`：planner 输出给 answer 层消费的压缩证据

### 数据对象命名

- `paper`：论文
- `record`：`manifest.jsonl` 中的论文记录
- `block`：MinerU 清洗后的结构块
- `chunk`：面向检索的文本窗口
- `ChunkDocument`：chunk 的运行时检索对象
- `BM25Document`：BM25 内部打分对象
- 外层变量优先使用 `chunk_documents`，避免用 `documents` 混淆论文和检索窗口
- scope 过滤参数使用 `allowed_chunk_ids`
- README 中的 document 只作为 `ChunkDocument` / `BM25Document` 术语出现，不表示论文

## 本地数据处理

### 数据导入

入库主流程：

1. 扫描 `PDF_DIR`，默认是 `data/pdf/`
2. 计算 PDF hash，用 `data/manifest.jsonl` 维护论文状态
3. 对新增或需要刷新的 PDF 调用 MinerU，得到原始解析目录
4. 从 MinerU 输出提取项目内部结构：`metadata`、`toc`、`blocks`、`chunks`、`references`
5. 查询外部元数据源，补全 title、authors、year、venue
6. 写回 `data/paper_data/<paper_id>/` 和 `manifest.jsonl`
7. 更新 `paper_annotations.json`
8. 全量同步末尾构建 `citation_graph.json`

`manifest` 负责记录本地论文库的状态，包括 `active`、`deleted`、`duplicate`、`error` 等

每条 active record 通常包含 `file_hash`、PDF 路径、title、authors、year、venue、paper_data_path

### 元数据检索

元数据补全位于 `paper_rag/ingest/metadata_sources/`

以 MinerU 抽取出的标题作为锚点，用标题精确匹配去外部源补全论文 metadata，拒绝模糊匹配

- `arxiv.py`：根据标题做 ArXiv 精确匹配，提供 preprint 信息

  ```python
  class ArxivMatch:
      title: str
      authors: list[str]
      preprint_year: int
      venue: str
  ```

- `dblp.py`：根据标题做 DBLP 精确匹配，提供正式发表信息

  ```python
  class DblpMatch:
      title: str
      authors: list[str]
      year: int
      venue: str
  ```

- `semantic_scholar.py`：补充正式发表信息

  ```python
  class SemanticScholarMatch:
      title: str
      authors: list[str]
      year: int
      venue: str
  ```

- `retry.py`：统一外部请求重试、延迟和 429/timeout 处理

按 `ArXiv -> DBLP -> Semantic Scholar` 的顺序补全

`publish_year`优先使用 venue 字符串中明确的四位年份，其次使用 DBLP/Semantic Scholar 返回的年份

作者名在 ingest 合并层清洗，例如删除 DBLP 作者末尾的消歧编号：`Yu Qiao 0001 -> Yu Qiao`

最终合并结果

```python
class MetadataMatch:
    title: str
    authors: list[str]
    year: dict[str, int | None]
    venue: str | None
```

### MinerU 识别与清洗

`mineru.py` 封装 MinerU API 的上传、任务轮询、结果下载和解压

`extract.py` 负责把 MinerU 的原始输出转换成项目内部稳定格式

主要输入通常来自 MinerU 输出中的 `content_list_v2.json`

清洗阶段会处理：

- 页面级内容展平
- 正文、标题、表格、图片等 block 文本抽取
- HTML table 转半结构化文本
- abstract、references、appendix、acknowledgement 等区域边界识别
- 目录树 `toc.json` 构建
- 原始 references 抽取到 `references.jsonl`

MinerU 原始结果保留在 `data/mineru_output/`，项目内部数据写入 `data/paper_data/<paper_id>/`

#### MinerU介绍

MinerU框架

- 预处理：使用 PyMuPDF 读取 PDF 文件，判断是否可解析、是否扫描、是否乱码、语言、页面尺寸
- 内容解析：采用 PDF 文档提取算法库 `PDF-Extract-Kit`，将不同的识别器应用于不同的区域，得到页面的语义 block 边界
- 后处理：根据第二阶段的输出，处理 bbox 包含/重叠，排序，删除无效区域
- 格式转换：生成中间 JSON、Markdown、最终 JSON

**预处理**

解析页面元数据，包括语言类型、总页数、页面尺寸

PDF 常见三类情况：

- Born-digital PDF：原生 LaTeX/Word 生成，文字层存在，可以直接用 PyMuPDF 抽文本
- 扫描 PDF：本质是图片，没有可靠文字层，需要启用 OCR
- 乱码 PDF：有文字层，但复制出来是 CID/乱码，直接抽文本会污染后续分割，需要提前进行识别，后续使用OCR进行文字识别

**内容解析**

不是先读出所有文字再猜段落，而是先在页面图像上检测不同区域

layout 标注类型包括标题、正文段落、图像、图像说明、表格、表说明、内联公式、公式标签和丢弃类型(页眉、页脚、页码和页注释)

先得到每个区域的 `bbox` 和类别，再基于几何关系重建阅读顺序

避免了直接 OCR 或直接抽 PDF 文本，公式变成乱码进而破坏句子、段落和 token 边界

`PDF-Extract-Kit`提到把任务拆成 layout detection、formula detection、formula recognition、OCR、table recognition 等模块

**后处理**

把整页切成若干 region，每个 region 最多包含一栏，这样可以保证文本在每个 region 内逐行自上而下读取

再根据 region 的位置关系排序，确定 PDF 中每个元素的阅读顺序

#### 数据清洗

总体流程如下

```
读取 MinerU JSON
  -> 扁平化，把二维结构 page -> blocks 变成一维
  -> 区域识别
  -> 目录识别
  -> 正文 block 输出
  -> chunk 输出
  -> reference 输出
  -> 落盘
```

特殊内容：

- 图片本身不做OCR，只使用 caption 作为检索文本

- 表格将识别出来的 HTML 转成半结构化文本

  举例：

  ```
  Table: Results on ImageNet.
  Columns: Method, Top-1, Top-5.
  Row 1: Method = ResNet-50; Top-1 = 76.1; Top-5 = 92.9.
  Row 2: Method = ViT-B; Top-1 = 81.8; Top-5 = 95.6.
  ```

**标题识别**

1. 找所有 `type == "title"` 且有文本的 block
2. 找 abstract marker
3. 优先取 abstract 之前的第一个非特殊 title
4. 找不到，就退回第一页 `page_header`

**区域边界识别**

1. `Abstract` 边界：找 title 型 abstract，如果没有找段落开头是`Abstract:`的 paragraph

2. `References` 边界：找第一个 title 型 References

3. `Appendix` 边界：兼容两种论文结构

   ```
   正文 -> Appendix -> References
   正文 -> References -> Appendix
   ```

4. `Acknowledgement` 边界：识别在 references 前、abstract 后的，作为正文结束的边界之一

5. `body` 边界：在摘要之后，references / appendix / acknowledgement 之前，找正文内容标题

   如果有编号标题，优先取第一个带编号标题作为正文起点；否则退回第一个有效标题

**区域判定**

把每个 `FlatBlock` 标成

```
abstract
body
appendix
reference
None
```

```
abstract_start <= idx < abstract_end       -> abstract
appendix_before_ref <= idx < references    -> appendix
body_start <= idx < body_end               -> body
idx >= appendix_after_ref                  -> appendix
idx > references_start                     -> reference
```

**TOC 构建**

从正文标题生成两套结构

```
sections: 扁平 section 列表
tree:     树形 toc
```

正文区域，如果整篇正文里存在编号标题，那么未编号标题会被跳过

Appendix区域，创建一个整体 appendix section

**References输出**

只从 references 区域的 reference_list block 抽取原始引用证据

抽出每条引用的 `raw_text`，输出格式是：

```
{
    "reference_id": "ref_001",
    "ref_index": 1,
    "raw_text": "...",
    "page": 12,
    "source_block_id": "b000456"
}
```

支持两类编号格式 `[1]` / `(1)`

#### block生成

生成最终的 `blocks.jsonl`，只保留后续检索和证据定位需要的字段

```
{
    "block_id": "b000123",
    "order": 57,
    "region": "body",
    "type": "paragraph",
    "text": "...",
    "page": 4,
    "bbox": [...],
    "section_id": "sec_2_1",
    "section_path": ["2 Method", "2.1 Architecture"]
}
```

### Chunk 生成

chunk 是面向检索的文本窗口

设置参数

```
DEFAULT_CHUNK_TARGET_CHARS = 1400  # chunk 最长长度
DEFAULT_CHUNK_OVERLAP_CHARS = 200  # chunk 重叠部分长度
MAX_CHUNK_EQUATION_CHARS = 500     # 单个 chunk 中允许保留的公式文本总字符数上限
```

每组 section blocks 内部，按 `target_chars` 累积

在组装 chunk 时，overlap 只进入 embedding_text，不进入 text

```
text           给用户展示 / 证据引用
embedding_text 给向量化使用，带 overlap 上下文
```

并且embedding_text 加 Paper 和 Section 前缀，可以提升跨论文检索的可辨性

### Citation Graph

在 ingest 全量同步末尾生成：

```text
data/paper_data/citation_graph.json
```

边方向固定为：

```text
source -> target
```

- `source`：引用发出论文
- `target`：被引用的本地论文

citation graph 只覆盖本地 active 论文之间的引用关系

匹配条件同时满足：

- target canonical title 出现在 reference raw text 的 normalized 文本中
- target 第一作者姓氏出现在 reference raw text token 中
- reference raw text 中出现 target 年份候选之一

避免短标题或常见词造成误配，年份候选包括 `preprint_year`、`publish_year`

`references.jsonl` 保留原始引用证据，`citation_graph.json` 是派生索引

### 别名与标签

`ingest/annotations.py` 管理 `data/paper_annotations.json`

该文件用于人工扩展论文别名和标签

人工维护：

- `aliases`：论文简称、常用名、大小写变体等
- `tags`：面向语义召回的人工标签

其它字段应由 ingest/API 生成，避免人工编辑与自动流程冲突

`corpus/venues.py` 管理 `data/venue_aliases.json`，使用 `canonical / display / aliases` 三层设计，匹配时使用 canonical 和 aliases，展示时使用 display

## Corpus

`corpus/` 是本地数据执行层，负责把 parser 输出落到本地数据上

主要职责：

- 读取 active manifest records
- 解析 paper mention、venue alias、year interval
- 按 metadata filters 收缩论文范围
- 加载 ChunkDocument 并按论文范围过滤
- 读取 annotation aliases/tags
- 读取 citation graph 并计算 follow/prior 范围
- 加载或构建 BM25 corpus index

filter 合法组合固定为：

- `paper`: `=` / `follow` / `prior`
- `year`: `=` / `interval`
- `venue`: `=` / `in`
- `author`: `contains`
- `title`: `contains`

禁止组合包括：

- `paper in`
- `year contains`
- `author =`
- `title =`
- `venue contains`
- `follow/prior` 用在 `paper` 以外字段

`filters` 数组内多个条件默认是 AND

OR/PER/AND 分组通过 `paper_groups + group_mode` 表示，不用多个同字段 `=` filter 表达 OR

运行期会用 `CorpusContext` 作为单次 `plan/ask` 的本地数据上下文

它会 lazy load manifest、annotations、chunks、citation graph 和 BM25 index

主要避免同一次 plan 内 router 规范化、planner 执行、多 group scope、reference source/object scope 和 content BM25 检索重复读取本地 JSON/JSONL

## 检索

检索链路：

```text
query
  -> top parser
  -> metadata/reference/content router
  -> domain planner
  -> evidence
```

顶层 parser 只做路由分类：

```json
{"router": "metadata|reference|content|unclear"}
```

具体语义解析交给三条 domain parser

### Common

`retrieval/routes/common/` 放 parser 共享能力：

- `parser_client.py`：OpenAI-compatible chat completion client
- `schema.py`：paper filters、groups、nullable enum 等共享校验
- `prompt.py`：共享 prompt 片段
- `errors.py`：`PlanParseError`

### Metadata

metadata route 回答论文元字段问题，例如作者、年份、venue、标题、数量、存在性

典型问题：

```powershell
python -m paper_rag ask "ResNet 和 Transformer 分别是哪一年发表的？"
python -m paper_rag ask "发表在 CVPR 上的论文有哪些？"
python -m paper_rag ask "2018 年的论文有多少篇？"
```

Schema 字段：

```json
{
  "intent": "lookup|list|count|exists|null",
  "return_fields": ["author|year|venue|title"],
  "paper_semantic": "",
  "filters": [],
  "paper_groups": [{"semantic": "", "filters": []}],
  "group_mode": "single|per|or|and"
}
```

- `lookup` 必须有 `return_fields`
- `list` 没有 `return_fields` 时默认返回 `title`
- `count/exists/null` 要求 `return_fields=[]`
- `group_mode="and"` 只允许用于 `exists`

执行时，`router.py` 把 parser result 规范化成 `RouteDecision`

通过查询 manifest records，再构建 metadata evidence

### Reference

reference route 回答本地库内 citation graph 覆盖的引用关系问题

统一语义是：

```text
source_scope --cites--> object_scope
```

典型问题：

```powershell
python -m paper_rag ask "ResNet 引用了哪些论文？"
python -m paper_rag ask "哪些论文引用了 ResNet？"
python -m paper_rag ask "哪些论文同时引用了 Transformer 和 ResNet？"
```

Schema 字段：

```json
{
  "intent": "list|count|exists|null",
  "return_side": "source|object|null",
  "source_semantic": "",
  "source_filters": [],
  "source_groups": [{"semantic": "", "filters": []}],
  "source_mode": "single|per|or|and",
  "object_semantic": "",
  "object_filters": [],
  "object_groups": [{"semantic": "", "filters": []}],
  "object_mode": "single|per|or|and"
}
```

- `list/count` 要求 `return_side=source|object`
- `exists/null` 要求 `return_side=null`
- `return_side="source"` 返回引用发出方论文
- `return_side="object"` 返回被引用方论文
- 执行层优先使用本地 `citation_graph.json`，图缺失时返回 `status="graph_missing"` 和 warning
- source/object 两侧的 filters 都先经过 `parser_scope_resolver` 标准化，再用 `paper_scope_records` 得到候选论文集合

### Content

content route 回答论文正文内容问题，例如方法、结构、实验、对比、原因、结论

典型问题：

```powershell
python -m paper_rag ask "ResNet 的模型结构是什么？"
python -m paper_rag ask "ResNet 里的 BasicBlock 和 Bottleneck 有什么区别？"
python -m paper_rag ask "ViT 使用了哪些数据集？"
```

Schema 字段：

```json
{
  "intent": "lookup|reason|compare|summary|list|count|exists|null",
  "paper_semantic": "",
  "filters": [],
  "paper_groups": [{"semantic": "", "filters": []}],
  "group_mode": "single|per|or|and",
  "content_objects": [],
  "compare_objects": []
}
```

content 检索顺序：

1. 用 semantic、filters、groups 解析候选论文范围
2. 如果候选论文为空，直接返回空 evidence
3. 构建 `dense_query` 和 `bm25_queries`
4. 只加载并保留候选论文的 chunks
5. 分别执行 dense 和 BM25
6. 用 RRF 融合命中 chunks
7. 对命中 chunk 扩展 block 上下文窗口
8. 构建 content evidence，交给回答 LLM

#### Dense

dense 检索位于 `retrieval/dense/`

- `embedding.py` 调用 OpenAI-compatible `/embeddings` 接口
- `cache.py` 按 model、dimensions 和文本缓存 embedding
- `milvus_store.py` 管理 Milvus/Zilliz collection
- `service.py` 提供 `run_index()`、`run_search()`、`search_dense_chunks()`

`dense_query` 保持中文自然语言语义句，不拼接年份、venue、论文标题等已经结构化的 scope 条件

这样做的目的是把论文范围过滤交给本地结构化层，把语义相似度交给 embedding 层

content route 执行 dense 检索时，会把候选论文的 `paper_id` 传给 Milvus filter

query embedding 使用独立的 `QUERY_EMBEDDING_CACHE_PATH`

#### BM25

BM25 位于 `retrieval/sparse/bm25.py`，用于对英文论文正文 chunks 做关键词召回

`paper-rag index` 会从 ChunkDocument 构建 `BM25CorpusIndex`

派生索引写入 `data/index/bm25_chunks.json`

落盘内容主要是：

- `doc_id`：实际对应 `chunk_id`
- `text_hash`：用于判断当前 chunk 文本是否变化
- `tokens`：`chunk.text + chunk.embedding_text` 规范化后的 token 列表

它不直接保存 BM25 分数、词频表、chunk 长度或 IDF

查询阶段优先加载这个派生索引，缺失或过期时退回内存构建

加载后会在内存中基于 tokens 计算：

- `f(t,d)`：token 在当前 chunk 中的出现次数
- `|d|`：当前 chunk 的 token 数
- 当前 scope 内的 `N`、`df(t)`、`avgdl`

派生索引有效时，不会重新 tokenize 全部 chunk

query 本身仍会在每次检索时 tokenize

`bm25_queries` 偏关键词候选，来源包括：

- `content_objects`
- `compare_objects`
- 从 query 剩余文本中抽取的核心词
- 翻译得到的英文候选

IDF：

```math
\operatorname{idf}(t)
=
\log\left(
1+
\frac{N-df(t)+0.5}{df(t)+0.5}
\right)

```

- 如果一个 token 出现在很多候选 chunk 中，说明它很常见，区分能力弱，IDF 较低
- 如果一个 token 只出现在少数候选 chunk 中，说明它更稀有，区分能力强，IDF 较高

词频归一化：

```math
\operatorname{tf\_norm}(t,d)
=
\frac{
f(t,d)(k_1+1)
}{
f(t,d)
+
k_1\left(
1-b+b\frac{|d|}{\operatorname{avgdl}}
\right)
}
```

- 词频归一化项衡量当前 token 在 chunk 中的匹配强度

单个 query 对单个 chunk 的 BM25 分数：

```math
\operatorname{BM25}(q,d)
=
\sum_{t\in \operatorname{tokens}(q)}
\log\left(
1+
\frac{N-\operatorname{df}(t)+0.5}
{\operatorname{df}(t)+0.5}
\right)
\cdot
\frac{
f(t,d)(k_1+1)
}{
f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)
}
```

- `q`：用户 query，经 `normalize_bm25_token` 转成 token 列表
- `d`：当前 chunk，`|d|` 表示当前 chunk 的 token 数
- `N`：候选 scope 内的 chunk 数
- `df(t)`：候选 scope 内包含 token `t` 的 chunk 数
- `f(t, d)`：token `t` 在当前 chunk `d` 中出现次数
- `avgdl`：候选 scope 内 chunks 的平均 token 数
- `k1`：控制词频饱和，项目中使用 1.5
- `b`：控制 chunk 长度归一化强度，项目中使用 0.75

`N`、`df(t)` 和 `avgdl` 都按当前候选论文 scope 内的 chunks 计算

即使派生索引覆盖全库，BM25 也不是先全库打分再过滤

它会先用 `allowed_chunk_ids` 收缩候选 chunk，再计算当前 scope 的统计量和分数

BM25Document 的 `text` 来自 `chunk.text + chunk.embedding_text`

如果翻译服务未配置或调用失败，BM25 会退回原关键词，不影响 dense 检索

多个 `bm25_queries` 分别检索后先用 RRF 合并为 BM25 候选

```math
\operatorname{RRF}(d)
=
\sum_{i=1}^{m}
\frac{1}{k+\operatorname{rank}_i(d)}
```

RRF 的 k 系数项目中使用 60

dense 与 BM25 候选再做 RRF 融合

## 答案生成

`paper_rag/answer/service.py` 是 ask 总编排：

```text
query
  -> run_plan()
  -> evidence
  -> local answer 或 LLM answer
  -> answer payload
```

### Evidence

`retrieval/evidence.py` 负责把各 route planner 的执行结果整理成统一 evidence

默认 composer evidence 保留回答需要的精简字段，例如：

```json
{
  "query": "...",
  "route": "metadata|reference|content",
  "status": "ok",
  "intent": "...",
  "plan": {},
  "results": {},
  "warnings": []
}
```

压缩原则：

- 空数组、空对象、空字符串字段默认不输出
- `resolved` 默认不输出；只有 alias 命中或必要消歧信息时输出
- 完整 parser result、RouteDecision、records、raw edges、context units、retrieval source terms 只进 `debug`
- `--debug` 下会额外输出 `debug.timings_ms`，用于观察 parser、scope、dense、BM25、fusion、answer 等阶段耗时
- metadata/reference/content 分别输出适合回答层消费的 results 结构

### Local

`answer/local.py` 负责本地确定性回答

metadata/reference route 默认走 local answer，因为这些问题可以从本地结构化数据确定性组织答案：

- metadata：lookup/list/count/exists
- reference：list/count/exists，以及 source/object 两侧引用关系
- failure/empty 状态：输出清晰的本地失败原因或空结果说明

local answer 的目标是尽量不引入 LLM 幻觉，把本地 evidence 中已经确定的事实组织成人类可读文本

### LLM

`answer/llm.py` 负责 content route 的回答生成

content route 的结果通常是一组正文上下文，需要由回答模型综合组织，因此默认走 LLM answer

回答 prompt 输入的是压缩后的 evidence，而不是完整数据库记录

## 设计取舍

### 数据分层

项目把原始 PDF、MinerU 原始输出、项目内部结构化数据分开保存

这样可以在不重新上传 PDF 的情况下重建 `paper_data`

也方便定位问题是来自 PDF 本身、MinerU 识别，还是项目清洗逻辑

### blocks 与 chunks 分开

`blocks.jsonl` 保留接近论文结构的证据单元

它用于页面、bbox、section、上下文窗口和 evidence 定位

`chunks.jsonl` 是检索窗口

它服务 dense/BM25 召回，可以带 overlap 和 embedding 前缀

两者分开后，检索粒度可以调整，证据定位结构不需要跟着变化

### references 单独处理

references 不进入正文 chunk

它们单独写入 `references.jsonl`，再派生 `citation_graph.json`

这样可以避免引用列表污染正文检索

也让 reference route 可以用本地 citation graph 做确定性回答

### 先路由再检索

系统先判断问题属于 metadata、reference 还是 content

metadata 和 reference 能用本地结构化数据确定回答，不需要盲目做正文召回

content 才进入 dense/BM25 召回和 LLM answer

这样可以减少无关检索，也降低 LLM 幻觉空间

### corpus 作为本地数据访问层

`corpus/` 不生成数据，也不规划回答

它负责把 manifest、annotations、venue aliases、chunks、citation graph 读成统一查询对象

retrieval 和 answer 不直接散落读取数据文件

这样 parser 输出、本地过滤和证据构建之间有稳定边界

运行期再用 `CorpusContext` 做单次 `plan/ask` 内复用

它主要服务 plan 内部的重复访问：

- router 规范化 parser scope
- planner 执行本地 scope
- 多个 group 分别解析论文范围
- reference route 同时解析 source/object 两侧
- content route 过滤 chunks 并执行 BM25 检索

这些阶段会共享同一份 manifest、annotations、chunks、citation graph 和 BM25 index

### 本地确定性回答优先

metadata/reference route 默认走 local answer

因为这些问题的答案来自 manifest 和 citation graph，可以确定性组织

content route 默认走 LLM answer

因为正文内容通常需要综合多个上下文片段，适合由模型组织自然语言答案

## 输出示例

### 端到端例子

以问题为例：

```powershell
python -m paper_rag ask "ResNet 里的 BasicBlock 和 Bottleneck 有什么区别？"
```

执行链路：

```text
query
  -> top parser
  -> content router
  -> content planner
  -> dense/BM25 retrieval
  -> chunk fusion
  -> context expansion
  -> evidence
  -> LLM answer
```

1. top parser 只判断问题类型，问题会进入 `content`

2. content parser 抽取正文检索结构

   得到：

   ```json
   {
     "intent": "compare",
     "paper_semantic": "",
     "filters": [{"field": "paper", "op": "=", "value": "ResNet"}],
     "paper_groups": [],
     "group_mode": "single",
     "content_objects": [],
     "compare_objects": ["BasicBlock", "Bottleneck"]
   }
   ```

3. router 把 parser result 规范化成 `RouteDecision`

   `paper=ResNet` 进入论文范围

   `BasicBlock` 和 `Bottleneck` 保留为正文比较对象

4. corpus 层解析论文范围

   解析 paper mention 和 aliases

   读取 active manifest records

   根据 filters 得到 ResNet 对应的论文 record

5. content planner 只加载候选论文的 chunks，不会对全库 chunks 盲检

   dense 检索也会把 scope 内的 `paper_id` 传给 Milvus filter

   BM25 会复用 `data/index/bm25_chunks.json` 或运行期内存索引

6. 构建检索 query

   `dense_query` 会变成适合 embedding 的自然语言句子，例如：

   ```
   比较 BasicBlock、Bottleneck 之间的差异和相关描述
   ```

   `bm25_queries` 会保留关键词候选，例如：

   ```
   ["BasicBlock", "Bottleneck"]
   ```

   如果翻译服务可用，还会补充英文候选

7. 执行 dense 和 BM25

   dense 在向量库中找语义相似 chunk

   BM25 在候选 chunk 文本中找关键词匹配

   BM25 的 IDF、avgdl 等统计量只按候选论文 chunks 计算

   多个 BM25 query 先用 RRF 合并成 BM25 候选

   dense 候选和 BM25 候选再用 RRF 合并成最终 chunk 排序

8. 扩展上下文

   命中 chunk 会根据 `block_ids` 回到 `blocks.jsonl`

   在同一 section 内扩展前后窗口

   最终形成 `context_units`

9. 构建 evidence

   默认 evidence 只保留回答需要的字段：

   - chunk id
   - 论文标题
   - section path
   - 页码
   - chunk text
   - expanded blocks

   完整 parser result、RouteDecision、原始 records、检索 source terms 只在 `--debug` 下输出

10. 生成答案

    把压缩后的 evidence 交给回答模型

    模型只基于 evidence 组织 BasicBlock 和 Bottleneck 的差异说明

### 失败路径例子

以 parser 配置缺失为例：

```powershell
python -m paper_rag ask "ResNet 的结构是什么？" --json
```

如果 `.env` 中缺少 `PLAN_PARSER_BASE_URL`、`PLAN_PARSER_API_KEY` 或 `PLAN_PARSER_MODEL`

执行链路会变成：

```text
query
  -> top parser
  -> parse_failed
  -> unclear evidence
  -> local answer
```

此时不会继续进入 metadata/reference/content planner

planner 会把失败原因放进 evidence：

```json
{
  "query": "ResNet 的结构是什么？",
  "route": "unclear",
  "status": "parse_failed",
  "results": {},
  "warnings": [
    "top_parse_failed: PLAN_PARSER_BASE_URL, PLAN_PARSER_API_KEY or PLAN_PARSER_MODEL is missing"
  ],
  "parser_error": "PLAN_PARSER_BASE_URL, PLAN_PARSER_API_KEY or PLAN_PARSER_MODEL is missing"
}
```

ask 层看到 `status != "ok"` 后不会调用回答 LLM

它会走本地失败回答：

```text
问题解析失败，暂时无法回答
```

这个路径用于保证 parser、配置或外部服务异常时，系统仍然返回可解释的失败结果

## 性能优化记录

以下结果来自本地链路测试

测试不调用真实 LLM、embedding API 或 Milvus 网络服务

因此它衡量的是本地可控开销，不代表完整 `ask` 端到端耗时

当前测试数据：

- 16 篇论文
- 767 个 chunks
- 放大测试使用 15340 个 chunks

### BM25 派生索引

优化点：

- `paper-rag index` 写出 `data/index/bm25_chunks.json`
- 派生索引保存 chunk 级 tokens 和 `text_hash`
- 查询阶段优先加载派生索引
- 当前 scope 内的 `N`、`df(t)`、`avgdl` 和 BM25 分数仍在查询时计算

收益来源：

- 避免每次 content 检索都对全库 chunks 重新 normalize/tokenize
- 避免每个 BM25 query 都重复构建 chunk tokens
- 保留 scope 内统计量，避免把全库 IDF 固化进索引

本地测试结果：

| 场景 | 旧路径 | 当前路径 | 提升 |
|---|---:|---:|---:|
| 实际库，4 个 BM25 query | 542 ms | 191 ms | 约 2.8x |
| 实际库，内存 index 复用 | 542 ms | 126 ms | 约 4.3x |
| 15340 chunks，4 个 BM25 query | 10014 ms | 3931 ms | 约 2.5x |
| 15340 chunks，内存 index 复用 | 10014 ms | 2566 ms | 约 3.9x |

### CorpusContext 复用

优化点：

- 单次 `plan/ask` 内创建一个 `CorpusContext`
- lazy load manifest、annotations、chunks、citation graph 和 BM25 index
- router、planner、scope resolver 和 content planner 共享同一份本地数据对象

收益来源：

- 避免同一次 plan 内重复读取 JSON/JSONL
- 避免多 group scope 重复构建本地对象
- 避免 reference source/object 两侧重复加载 citation graph
- 避免 content route 过滤 chunks 和 BM25 检索时重复读 chunks

本地测试结果：

| 场景 | 旧路径 | 当前路径 | 提升 |
|---|---:|---:|---:|
| repeated scope 查询 | 75 ms | 21 ms | 约 3.5x |

### Embedding Cache

优化点：

- chunk embedding cache 和 query embedding cache 分开
- `EMBEDDING_CACHE_PATH` 主要服务 `index`
- `QUERY_EMBEDDING_CACHE_PATH` 主要服务 `search/plan/ask`
- cache 更新从全量重写改为 append-only

收益来源：

- 查询阶段不需要加载 chunk embedding 的大 cache 文件
- 重复 query 可以直接复用 query embedding
- 新增 cache 行时不再重写整个 JSONL 文件
- append-only 文件中如果出现重复 key，加载时以后出现的值为准

本地测试结果：

| 场景 | 旧路径 | 当前路径 | 提升 |
|---|---:|---:|---:|
| cache 5000 条，新增 1 条 | 55 ms | 28 ms | 约 2.0x |
| cache 20000 条，新增 1 条 | 219 ms | 118 ms | 约 1.9x |
| query cache 加载，混合大文件 vs 独立小文件 | 28 ms | 1.5 ms | 约 18x |

### 结果边界

- BM25 优化主要减少预处理成本，不改变 BM25 公式和排序语义
- CorpusContext 优化主要减少本地数据重复读取，不改变 scope 解析结果
- Embedding cache 优化主要减少缓存 I/O 和重复 embedding 请求，不改变 embedding 结果
- 完整 `ask` 耗时仍受 parser、embedding 服务、Milvus/Zilliz 和 answer LLM 影响

## 当前边界

- Reference 只回答本地 citation graph 覆盖的库内引用关系，不回答全网 citation network
- Citation graph 匹配规则偏保守，优先降低误配，可能漏掉格式异常或标题不完整的引用
- Content 的最终回答质量依赖 MinerU 正文解析质量、chunk 构建质量、dense/BM25 召回质量和回答 LLM
- BM25 主要面向英文论文正文；中文 query 依赖翻译候选增强英文关键词召回
- BM25 翻译未配置或调用失败时会退回原关键词，不影响 dense 检索
- `paper_annotations.json` 中的 aliases/tags 需要人工持续维护，才能提升简称、别名和语义标签召回
- Top parser 只做路由分类；如果 parser 配置缺失或返回 `unclear`，不会进入 domain planner
- Dense 检索依赖 embedding 服务和 Milvus/Zilliz；未配置时 `index/search/content` dense 链路不可用
