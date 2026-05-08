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
└─ chunk_fusion.py                   # dense/BM25 命中结果的 RRF 融合
```

```text
paper_rag/retrieval/dense/
├─ __init__.py                       # dense 子包标记
├─ service.py                        # index/search/content dense search 高层服务
├─ embedding.py                      # OpenAI-compatible embedding HTTP 客户端
├─ cache.py                          # embedding 本地缓存
└─ milvus_store.py                   # Milvus/Zilliz collection 重建、插入、向量搜索

paper_rag/retrieval/sparse/
├─ __init__.py                       # sparse 子包标记
└─ bm25.py                           # BM25 索引、英文 token 规范化、多 query RRF 合并
```

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

根据 `data/paper_data/*/chunks.jsonl` 重建 dense 向量索引

只处理向量库，不重新解析 PDF，不刷新 metadata

主要职责：

- 新增或刷新论文后同步 dense index
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
- `--debug`：输出中间状态

### ask

面向最终使用的问答入口

回答策略：

- `metadata`：本地确定性回答
- `reference`：本地 citation graph 回答
- `content`：基于 evidence 调用回答 LLM
- `unclear` 或无证据：返回降级说明

参数：

- `<query>`：要回答的问题
- `--debug`：在 answer payload 中保留 planner debug 信息
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
    source: str  # 信息检索来源
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
- 加载 chunk 文档并按论文范围过滤
- 读取 annotation aliases/tags
- 读取 citation graph 并计算 follow/prior 范围

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

metadata route 回答论文元字段问题，例如作者、年份、venue、标题、数量、存在性。

典型问题：

```powershell
python -m paper_rag ask "ResNet 和 Transformer 分别是哪一年发表的？"
python -m paper_rag ask "发表在 CVPR 上的论文有哪些？"
python -m paper_rag ask "2018 年的论文有多少篇？"
```

metadata parser 输出的核心字段：

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

执行时，`router.py` 把 parser result 规范化成 `RouteDecision`，`planner.py` 通过 `paper_scope_records.records_for_scope()` 查询 manifest records，再构建 metadata evidence。

### Reference

reference route 回答本地库内 citation graph 覆盖的引用关系问题。统一语义是：

```text
source_scope --cites--> object_scope
```

典型问题：

```powershell
python -m paper_rag ask "ResNet 引用了哪些论文？"
python -m paper_rag ask "哪些论文引用了 ResNet？"
python -m paper_rag ask "哪些论文同时引用了 Transformer 和 ResNet？"
```

reference parser 输出的核心字段：

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

`return_side="source"` 返回引用发出方，`return_side="object"` 返回被引用方。planner 优先使用 `data/paper_data/citation_graph.json`，如果图缺失会返回 `graph_missing` warning，不临时联网，也不扫描全库兜底。

### Content

content route 回答论文正文内容问题，例如方法、结构、实验、对比、原因、结论。

典型问题：

```powershell
python -m paper_rag ask "ResNet 的模型结构是什么？"
python -m paper_rag ask "ResNet 里的 BasicBlock 和 Bottleneck 有什么区别？"
python -m paper_rag ask "ViT 使用了哪些数据集？"
```

content parser 输出的核心字段：

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

1. 用 semantic、filters、groups 解析候选论文范围。
2. 只保留候选论文的 chunks。
3. 构建 `dense_query` 和 `bm25_queries`。
4. 分别执行 dense 和 BM25。
5. 用 RRF 融合命中 chunks。
6. 对命中 chunk 扩展 block 上下文窗口。
7. 构建 content evidence，交给回答 LLM。

#### Dense

dense 检索位于 `retrieval/dense/`。

- `embedding.py` 调用 OpenAI-compatible `/embeddings` 接口。
- `cache.py` 按 model、dimensions 和文本缓存 embedding。
- `milvus_store.py` 管理 Milvus/Zilliz collection。
- `service.py` 提供 `run_index()`、`run_search()`、`search_dense_chunks()`。

`dense_query` 保持中文自然语言语义句，不拼接年份、venue、论文标题等已经结构化的 scope 条件。这样做的目的是把论文范围过滤交给本地结构化层，把语义相似度交给 embedding 层。

#### BM25

BM25 位于 `retrieval/sparse/bm25.py`，用于对英文论文正文 chunks 做关键词召回。

`bm25_queries` 偏关键词候选，来源包括：

- `content_objects`
- `compare_objects`
- 从 query 剩余文本中抽取的核心词
- 腾讯/阿里翻译得到的英文候选

如果翻译服务未配置或调用失败，BM25 会退回原关键词，不影响 dense 检索。dense 与 BM25 的结果通过 `chunk_fusion.py` 做 RRF 融合。

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

`retrieval/evidence.py` 负责把各 route planner 的执行结果整理成统一 evidence。

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

- 空数组、空对象、空字符串字段默认不输出。
- `resolved` 默认不输出；只有 alias 命中或必要消歧信息时输出。
- 完整 parser result、RouteDecision、records、raw edges、context units、retrieval source terms 只进 `debug`。
- metadata/reference/content 分别输出适合回答层消费的 results 结构。

### Local

`answer/local.py` 负责本地确定性回答。

metadata/reference route 默认走 local answer，因为这些问题可以从本地结构化数据确定性组织答案：

- metadata：lookup/list/count/exists。
- reference：list/count/exists，以及 source/object 两侧引用关系。
- failure/empty 状态：输出清晰的本地失败原因或空结果说明。

local answer 的目标是尽量不引入 LLM 幻觉，把本地 evidence 中已经确定的事实组织成人类可读文本。

### LLM

`answer/llm.py` 负责 content route 的回答生成。

content route 的结果通常是一组正文上下文，需要由回答模型综合组织，因此默认走 LLM answer。回答 prompt 输入的是压缩后的 evidence，而不是完整数据库记录。

回答 LLM 使用 OpenAI-compatible chat completion 接口。配置来自 `ANSWER_*`，未配置时回退到 `PLAN_PARSER_*`。

## 当前边界

- Reference 只回答本地 citation graph 覆盖的库内引用关系，不回答全网 citation network。
- Citation graph 匹配规则偏保守，优先降低误配，可能漏掉格式异常或标题不完整的引用。
- Content 的最终回答质量依赖 MinerU 正文解析质量、chunk 构建质量、dense/BM25 召回质量和回答 LLM。
- BM25 主要面向英文论文正文；中文 query 依赖翻译候选增强英文关键词召回。
- BM25 翻译未配置或调用失败时会退回原关键词，不影响 dense 检索。
- `paper_annotations.json` 中的 aliases/tags 需要人工持续维护，才能提升简称、别名和语义标签召回。
- Top parser 只做路由分类；如果 parser 配置缺失或返回 `unclear`，不会进入 domain planner。
- Dense 检索依赖 embedding 服务和 Milvus/Zilliz；未配置时 `index/search/content` dense 链路不可用。
- 当前 README 记录的是主链路设计，后续可以在各节继续补充关键函数代码片段和示例 payload。
