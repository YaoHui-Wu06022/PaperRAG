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

`data/` 的约定结构：

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

`paper_rag/` 的模块职责：

```text
paper_rag/
├─ __init__.py                       # 包标记
├─ __main__.py                       # `python -m paper_rag` 入口
├─ config.py                         # Settings 与 .env 读取、路径解析、默认配置
├─ utils.py                          # 根包通用工具：hash、slug、文本规范化、安全目录替换
├─ answer/
│  ├─ __init__.py                    # answer 包入口，re-export run_ask
│  ├─ service.py                     # ask 薄编排：plan evidence -> local/LLM answer
│  ├─ local.py                       # metadata/reference 的本地确定性回答
│  └─ llm.py                         # content route 的回答 LLM 客户端与 prompt 组装
├─ cli/
│  ├─ __init__.py                    # CLI 子包标记
│  ├─ main.py                        # CLI 总入口与全局参数
│  ├─ ingest.py                      # `paper-rag ingest`
│  ├─ retrieval.py                   # `paper-rag index/search/plan`
│  └─ ask.py                         # `paper-rag ask`
├─ ingest/
│  ├─ __init__.py                    # 入库子包标记
│  ├─ pipeline.py                    # 全量 PDF 同步、metadata 补全、extract、citation graph 主流程
│  ├─ manifest.py                    # ManifestRecord/Manifest、状态管理、年份规范化
│  ├─ mineru.py                      # MinerU API 上传、轮询、下载、解压
│  ├─ extract.py                     # 从 MinerU 输出构建 metadata/toc/blocks/references/chunks
│  ├─ citation_graph.py              # 从 references 构建本地库内 citation graph
│  ├─ annotations.py                 # paper_annotations.json 生成、规范化与保存
│  ├─ venues.py                      # venue canonical/display/aliases 规范化与匹配
│  └─ metadata_sources/
│     ├─ __init__.py                 # 元数据检索子包标记
│     ├─ arxiv.py                    # ArXiv 精确标题查询与 preprint metadata
│     ├─ dblp.py                     # DBLP 精确标题查询与正式发表 metadata
│     ├─ semantic_scholar.py         # Semantic Scholar 正式发表信息补充
│     └─ retry.py                    # 外部元数据请求的重试、退避、延迟
├─ corpus/
│  ├─ __init__.py                    # 本地结构化论文库访问层
│  ├─ aliases.py                     # 论文 mention/alias 到 canonical paper 的匹配
│  ├─ annotations.py                 # paper_annotations.json 的统一扫描入口
│  ├─ chunks.py                      # chunks.jsonl 加载、ChunkDocument、按论文过滤
│  ├─ citations.py                   # `paper follow/prior` 基于 citation graph 的范围解析
│  ├─ filters.py                     # 单条 manifest record 的 filter 布尔匹配
│  ├─ records.py                     # active manifest 读取、论文匹配、record key、去重
│  ├─ scope.py                       # semantic + filters + groups 到候选论文 records
│  ├─ resolver.py                    # parser 输出中的 paper/year/venue scope 标准化
│  └─ utils.py                       # token 规范化、去重、interval boundary、value 展平
└─ retrieval/
   ├─ __init__.py                    # 检索子包标记
   ├─ plan.py                        # `paper-rag plan` 编排：top route -> domain router -> planner
   ├─ route.py                       # RouteDecision，保存 parser 归一化后的路由状态
   ├─ evidence.py                    # composer/debug evidence 统一构建
   ├─ evidence_probe.py              # evidence 调试入口
   ├─ chunk_fusion.py                # dense/BM25 命中结果的 RRF 融合
   ├─ dense/
   │  ├─ __init__.py                 # dense 子包标记
   │  ├─ service.py                  # index/search/content dense search 高层服务
   │  ├─ embedding.py                # OpenAI-compatible embedding HTTP 客户端
   │  ├─ cache.py                    # embedding 本地缓存
   │  └─ milvus_store.py             # Milvus/Zilliz collection 重建、插入、向量搜索
   ├─ sparse/
   │  ├─ __init__.py                 # sparse 子包标记
   │  └─ bm25.py                     # BM25 索引、英文 token 规范化、多 query RRF 合并
   └─ routes/
      ├─ __init__.py                 # route 子包标记
      ├─ common/
      │  ├─ __init__.py              # common 子包标记
      │  ├─ errors.py                # PlanParseError
      │  ├─ parser_client.py         # OpenAI-compatible planner parser client
      │  ├─ prompt.py                # 三条 route 共用 prompt 片段
      │  └─ schema.py                # 三条 route 共用 schema/filter/group 校验
      ├─ top/
      │  ├─ __init__.py              # top route 子包标记
      │  ├─ parser.py                # 顶层路由 parser client
      │  ├─ prompt.py                # 顶层 route prompt，只分类 metadata/reference/content/unclear
      │  ├─ schema.py                # top schema: 只允许 router 字段
      │  └─ prompt_probe.py          # top prompt 调试入口
      ├─ metadata/
      │  ├─ __init__.py              # metadata route 子包标记
      │  ├─ parser.py                # metadata parser client
      │  ├─ prompt.py                # metadata parser prompt
      │  ├─ schema.py                # metadata parser 输出校验
      │  ├─ router.py                # parser_result -> RouteDecision
      │  ├─ planner.py               # metadata 本地查询执行与 evidence 组装
      │  ├─ prompt_probe.py          # metadata prompt 调试入口
      │  └─ planner_probe.py         # metadata planner 调试入口
      ├─ reference/
      │  ├─ __init__.py              # reference route 子包标记
      │  ├─ parser.py                # reference parser client
      │  ├─ prompt.py                # reference parser prompt
      │  ├─ schema.py                # reference parser 输出校验
      │  ├─ router.py                # source/object scope 修正与 RouteDecision 构建
      │  ├─ planner.py               # citation graph 查询执行与 evidence 组装
      │  ├─ prompt_probe.py          # reference prompt 调试入口
      │  └─ planner_probe.py         # reference planner 调试入口
      └─ content/
         ├─ __init__.py              # content route 子包标记
         ├─ parser.py                # content parser client
         ├─ prompt.py                # content parser prompt
         ├─ schema.py                # content parser 输出校验
         ├─ router.py                # content scope 修正与 RouteDecision 构建
         ├─ planner.py               # content 检索执行、融合、上下文扩展与 evidence 组装
         ├─ retrieval_query.py       # dense_query / bm25_queries 构建
         ├─ context.py               # chunk 命中后扩展 block 窗口
         ├─ translation.py           # BM25 中文关键词翻译：腾讯/阿里
         ├─ prompt_probe.py          # content prompt 调试，并可写入 retrieval_probe_cases.json
         ├─ retrieval_probe.py       # 跳过 prompt，用固定 case 调试 dense/BM25/fusion 召回
         ├─ retrieval_probe_cases.json # content retrieval 调试样例
         └─ planner_probe.py         # content planner 调试入口
```

## 配置

推荐 Python 3.10+。本地开发安装：

```powershell
pip install -e .
```

复制 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
```

常用配置分组：

- MinerU：`MINERU_API_KEY`、`MINERU_API_BASE_URL`、`MINERU_MODEL_VERSION`、`MINERU_LANGUAGE`
- 外部元数据：`DBLP_DELAY_SECONDS`、`DBLP_CANDIDATE_LIMIT`、`SEMANTIC_SCHOLAR_API_KEY`、`SEMANTIC_SCHOLAR_DELAY_SECONDS`、`ARXIV_DELAY_SECONDS`
- Planner LLM：`PLAN_PARSER_BASE_URL`、`PLAN_PARSER_API_KEY`、`PLAN_PARSER_MODEL`、`PLAN_PARSER_TIMEOUT_SECONDS`
- Answer LLM：`ANSWER_BASE_URL`、`ANSWER_API_KEY`、`ANSWER_MODEL`、`ANSWER_TIMEOUT_SECONDS`、`ANSWER_TEMPERATURE`
- 目录：`PDF_DIR`、`MINERU_DIR`、`PAPER_DIR`
- Chunk：`CHUNK_TARGET_CHARS`、`CHUNK_OVERLAP_CHARS`
- Milvus/Zilliz：`MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_DB_NAME`、`MILVUS_COLLECTION`
- Embedding：`EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_BATCH_SIZE`、`EMBEDDING_CACHE_PATH`
- BM25 关键词翻译：`PLAN_BM25_TRANSLATE_PROVIDERS`、`TENCENT_TRANSLATE_*`、`ALIYUN_TRANSLATE_*`

`ANSWER_*` 默认可复用 `PLAN_PARSER_*`，即没有单独配置回答模型时，会回退到 planner parser 的 base URL、API key 和 model。

密钥不要提交到 Git。当前仓库忽略了 `.env` 和 `密匙/`，如需使用 `keys/`、`secrets/`、`密钥/` 等目录，也应同步加入 `.gitignore`。

Windows PowerShell 中文输出异常时可以先设置：

```powershell
$OutputEncoding=[System.Text.Encoding]::UTF8
[Console]::InputEncoding=[System.Text.Encoding]::UTF8
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING='utf-8'
```

## CLI

包安装后可使用 `paper-rag`，也可以直接使用 `python -m paper_rag`。两者入口相同。

全局参数：

```text
paper-rag [--project-root PROJECT_ROOT] <command>
```

子命令：

```text
paper-rag ingest [--refresh] [--quiet]
paper-rag index [--quiet]
paper-rag search [--top-k TOP_K] <query>
paper-rag plan [--debug] <query>
paper-rag ask [--debug] [--json] <query>
```

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

`plan` 输出给回答链路消费的 evidence。加 `--debug` 后会输出 parser result、scope、retrieval query、raw records、context units 等中间状态。

`ask` 会先执行 `run_plan()`，再根据 route 选择本地回答或 LLM 回答。`--json` 输出完整 payload，包括 `answer`、`answer_mode`、`evidence` 和 warnings。

## 命名约定

- `query`：用户原始问题，全链路统一使用这个名字。
- `route`：顶层语义路由，取值为 `metadata`、`reference`、`content`、`unclear`。
- `parser_result`：LLM parser 的结构化输出。
- `RouteDecision`：parser 结果经本地规范化后的不可变决策对象。
- `paper_semantic`：单侧论文范围的自然语言语义描述。
- `filters`：单侧论文范围的结构化过滤条件。
- `paper_groups` / `group_mode`：论文范围分组，用于表达 per/or/and 等关系。
- `source_*` / `object_*`：reference route 的两侧 scope；统一理解为 `source --cites--> object`。
- `retrieval_query`：content route 内部检索输入对象。
- `dense_query`：embedding 使用的中文自然语言语义句。
- `bm25_queries`：BM25 使用的关键词候选列表。
- `context_units`：content 检索命中 chunk 后扩展出的上下文单元。
- `evidence`：planner 输出给 answer 层消费的压缩证据。
- `debug`：只在调试模式下保留的内部状态。

通用 paper filter 的合法组合：

```text
paper:  = | follow | prior
year:   = | interval
venue:  = | in
author: contains
title:  contains
```

禁止组合包括 `paper in`、`year contains`、`author =`、`title =`、`venue contains`，以及把 `follow/prior` 用在 `paper` 以外字段。

## 本地数据处理

### 数据导入

入口是 `paper_rag/ingest/pipeline.py`，CLI 对应 `paper-rag ingest`。

入库主流程：

1. 扫描 `PDF_DIR`，默认是 `data/pdf/`。
2. 计算 PDF hash，用 `data/manifest.jsonl` 维护论文状态。
3. 对新增或需要刷新的 PDF 调用 MinerU，得到原始解析目录。
4. 从 MinerU 输出提取项目内部结构：`metadata`、`toc`、`blocks`、`chunks`、`references`。
5. 查询外部元数据源，补全 title、authors、year、venue。
6. 写回 `data/paper_data/<paper_id>/` 和 `manifest.jsonl`。
7. 更新 `paper_annotations.json`。
8. 全量同步末尾构建 `citation_graph.json`。

`manifest` 负责记录本地论文库的事实状态。状态包括 `active`、`deleted`、`duplicate`、`error` 等。每条 active record 通常包含 `file_hash`、PDF 路径、title、authors、year、venue、paper_data_path。

### 元数据检索

元数据补全位于 `paper_rag/ingest/metadata_sources/`。

- `arxiv.py`：根据标题做 ArXiv 精确匹配，主要补充 preprint 信息。
- `dblp.py`：根据标题做 DBLP 精确匹配，偏正式发表信息。
- `semantic_scholar.py`：补充 Semantic Scholar 中的正式发表信息。
- `retry.py`：统一外部请求重试、延迟和 429/timeout 处理。

合并规则偏保守：外部候选必须与论文标题精确归一化匹配；ArXiv 命中只写 `preprint_year`，不把 `venue` 写成 `ArXiv`；正式发表年份优先使用 venue 字符串中明确的四位年份，其次使用 DBLP/Semantic Scholar 返回的年份。

作者名在 ingest 合并层清洗，例如删除 DBLP 作者末尾的消歧编号：`Yu Qiao 0001 -> Yu Qiao`。

### MinerU 识别与清洗

`mineru.py` 封装 MinerU API 的上传、任务轮询、结果下载和解压。`extract.py` 负责把 MinerU 的原始输出转换成项目内部稳定格式。

主要输入通常来自 MinerU 输出中的 `content_list_v2.json`。清洗阶段会处理：

- 页面级内容展平。
- 正文、标题、表格、图片等 block 文本抽取。
- HTML table 转半结构化文本。
- abstract、references、appendix、acknowledgement 等区域边界识别。
- 目录树 `toc.json` 构建。
- 原始 references 抽取到 `references.jsonl`。

MinerU 原始结果保留在 `data/mineru_output/`，项目内部数据写入 `data/paper_data/<paper_id>/`。这使得解析过程可追溯，也便于未来重新清洗而不必重新上传 PDF。

### Block -> Chunk

`extract.py` 中的 block 是从 PDF 版面解析结果清洗得到的结构化单元，chunk 是面向检索的文本窗口。

block 层保留更多结构信息，例如页码、区域、section、block 类型、原始 source path、媒体字段等。chunk 层则面向检索，通常包含：

- `chunk_id`
- `paper_id`
- `chunk_index`
- `title`
- `section_path`
- `pages`
- `text`
- `embedding_text`
- chunk 覆盖的 block 范围

chunk 构建受 `CHUNK_TARGET_CHARS` 和 `CHUNK_OVERLAP_CHARS` 控制。`embedding_text` 会把标题、section path 和正文文本组合成更稳定的 embedding 输入。

### Citation Graph

`citation_graph.py` 在 ingest 全量同步末尾生成：

```text
data/paper_data/citation_graph.json
```

边方向固定为：

```text
source -> target
```

- `source`：引用发出论文。
- `target`：被引用的本地论文。

当前 citation graph 只覆盖本地 active 论文之间的引用关系。匹配规则偏保守，需要 reference raw text 同时命中目标论文 canonical title、第一作者姓氏和年份候选，避免短标题或常见词造成误配。

年份候选包括 `preprint_year`、`publish_year` 和 venue 字符串中的四位年份。`references.jsonl` 保留原始引用证据，`citation_graph.json` 是派生索引。

### 别名与标签

`annotations.py` 管理 `data/paper_annotations.json`。该文件用于人工扩展论文别名和标签。

当前建议人工维护：

- `aliases`：论文简称、常用名、大小写变体等。
- `tags`：面向语义召回的人工标签。

其它字段应由 ingest/API 生成，避免人工编辑与自动流程冲突。

`venues.py` 管理 `data/venue_aliases.json`，使用 `canonical / display / aliases` 三层设计：匹配时使用 canonical 和 aliases，展示时使用 display。

## 检索

检索链路从 `paper_rag/retrieval/plan.py` 开始：

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

它不抽 filters，不裁剪 query，也不生成 evidence。具体语义解析交给三条 domain parser。

### Common

`retrieval/routes/common/` 放 parser 共享能力：

- `parser_client.py`：OpenAI-compatible chat completion client。
- `schema.py`：paper filters、groups、nullable enum 等共享校验。
- `prompt.py`：共享 prompt 片段。
- `errors.py`：`PlanParseError`。

`corpus/` 是本地数据执行层，负责把 parser 输出落到本地数据上：

- `resolver.py`：解析 paper mention、venue alias、year interval。
- `scope.py`：把 semantic、filters、groups 转成候选论文 records。
- `records.py`：读取 active manifest、匹配论文、生成统一 record key。
- `filters.py`：单条 manifest record 的最终布尔匹配。
- `aliases.py`：结构化 paper mention 的别名匹配，不改写用户 query。
- `chunks.py`：加载 chunk 并按候选论文过滤。
- `citations.py`：基于 citation graph 处理 `paper follow/prior`。

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
