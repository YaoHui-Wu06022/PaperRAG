# Paper_RAG 当前计划与模块规范

## 0. 当前状态

- 当前 CLI：`paper-rag ingest`、`paper-rag index`、`paper-rag search`、`paper-rag plan`。
- `paper-rag ask` 当前不注册；后续等 answer composer 稳定后再接回。
- 用户输入统一称为 `query`；历史输入命名全部废弃。
- 顶层 parser 只做路由分类：`{"router": "metadata|reference|content|unclear"}`，不抽 filters，不裁剪 query，不生成 evidence。
- `paper-rag plan` 是薄编排：top route -> domain router -> domain planner -> 统一 evidence。
- 三条 route 的 planner 默认输出 composer 模式；`--debug` 才输出完整 parser/result/scope/retrieval 中间态。
- 代码注释默认使用中文；PowerShell 中文运行时统一设置 UTF-8。

## 1. 项目目标

把本地 PDF 论文库整理成可重建、可检索、可追溯的结构化知识库，并提供面向论文元数据、引用关系和正文内容的检索计划能力。

核心原则：

- 原始 PDF、MinerU 原始输出、项目内部 `paper_data` 分层保存。
- 数据处理阶段尽量沉淀可复用索引：manifest、metadata、chunks、references、citation graph、annotations。
- 检索阶段先解析语义和论文范围，再进入 metadata/reference/content 各自执行层。
- 对外 evidence 只保留回答组织需要的信息；内部路径、hash、完整 records、raw chunks 放在 debug。

## 2. 仓库结构与模块职责

```text
paper_rag/
├─ __main__.py                       # `python -m paper_rag` 入口
├─ config.py                         # Settings 与 .env 读取
├─ utils.py                          # 根包通用小工具
├─ cli/
│  ├─ main.py                        # CLI 总入口
│  ├─ ingest.py                      # `paper-rag ingest`
│  └─ retrieval.py                   # `index/search/plan`
├─ dataprocess/
│  ├─ ingest.py                      # 全量 PDF 同步、metadata、extract、citation graph 主流程
│  ├─ manifest.py                    # manifest 结构、读写、状态管理
│  ├─ mineru.py                      # MinerU API 上传、轮询、下载、解压
│  ├─ extract.py                     # 从 MinerU 输出构建 metadata/toc/blocks/references/chunks
│  ├─ citation_graph.py              # 本地 citation graph 构建
│  ├─ annotations.py                 # paper_annotations.json 生成与维护
│  ├─ venues.py                      # venue alias/display 规范化
│  └─ metadata/
│     ├─ arxiv.py                    # ArXiv 精确标题查询
│     ├─ dblp.py                     # DBLP 精确标题查询
│     ├─ semantic_scholar.py         # Semantic Scholar 正式发表信息补充
│     └─ retry.py                    # 外部请求重试/延迟
└─ retrieval/
   ├─ plan.py                        # `paper-rag plan` 薄编排
   ├─ route.py                       # RouteDecision
   ├─ evidence.py                    # composer/debug evidence 构建
   ├─ evidence_probe.py              # evidence 调试脚本
   ├─ chunk_fusion.py                # dense/BM25 RRF 融合
   ├─ data/
   │  ├─ aliases_match.py            # annotation aliases 到 canonical paper match
   │  ├─ annotations_index.py        # paper_annotations.json 统一扫描入口
   │  ├─ chunks_load.py              # chunks.jsonl 读取与按论文记录过滤
   │  ├─ citation_scope.py           # paper follow/prior citation 范围
   │  ├─ filters.py                  # manifest record filter evaluator
   │  ├─ manifest_records.py         # active manifest 读取、匹配、record key
   │  ├─ paper_scope_records.py      # semantic + filters + groups 到候选论文 records
   │  ├─ parser_scope_resolver.py    # parser scope/filter value 解析
   │  └─ utils.py                    # normalize token、dedupe、interval boundary 工具
   ├─ dense/
   │  ├─ service.py                  # index/search 接线与 dense chunk search
   │  ├─ embedding.py                # OpenAI-compatible embedding client
   │  ├─ cache.py                    # embedding cache
   │  └─ milvus_store.py             # Milvus/Zilliz collection 管理
   ├─ sparse/
   │  └─ bm25.py                     # BM25 index 与多 query RRF 合并
   └─ domains/
      ├─ common/
      │  ├─ errors.py                # PlanParseError
      │  ├─ parser_client.py         # OpenAI-compatible parser client
      │  ├─ schema.py                # 三路共用 schema/filter/group 校验
      │  └─ prompt.py                # 共用 prompt 片段
      ├─ top/
      │  ├─ parser.py                # top parser client
      │  ├─ prompt.py                # top route prompt
      │  ├─ prompt_probe.py          # top prompt 测试入口
      │  └─ schema.py                # top schema: router only
      ├─ metadata/
      │  ├─ parser.py / prompt.py / schema.py / router.py / planner.py
      │  ├─ prompt_probe.py
      │  └─ planner_probe.py
      ├─ reference/
      │  ├─ parser.py / prompt.py / schema.py / router.py / planner.py
      │  ├─ prompt_probe.py
      │  └─ planner_probe.py
      └─ content/
         ├─ parser.py / prompt.py / schema.py / router.py / planner.py
         ├─ retrieval_query.py       # dense_query / bm25_queries 组装
         ├─ context.py               # chunk 命中后扩 block 窗口
         ├─ translation.py           # BM25 关键词翻译：腾讯/阿里
         ├─ prompt_probe.py
         └─ planner_probe.py
```

## 3. 数据处理规范

- `data/pdf/` 是 PDF 输入目录。
- `data/manifest.jsonl` 记录 active/deleted/duplicate/error 状态、`file_hash`、PDF 路径、title、authors、year、venue、paper_data_path。
- `data/mineru_output/` 保存 MinerU 原始结果；`data/paper_data/<paper_id>/` 保存项目内部结构化结果。
- 每篇 active 论文的 `paper_data` 至少包含：
  - `metadata.json`
  - `toc.json`
  - `blocks.jsonl`
  - `chunks.jsonl`
  - `references.jsonl`
- `metadata.json` 的 `year` 固定为：
  ```json
  {"preprint_year": 2017, "publish_year": 2018}
  ```
- 正式发表年份优先规则：
  - 如果正式 `venue` 字符串含明确四位会议/期刊年份，`publish_year` 优先使用 venue 年份。
  - 否则使用 DBLP/Semantic Scholar 返回的 `year`。
  - ArXiv 命中只写 `preprint_year`，不把 `venue` 写成 `ArXiv`。
- 作者名在 ingest 合并层清洗，删除 DBLP 末尾消歧编号，例如 `Yu Qiao 0001 -> Yu Qiao`。
- `paper_annotations.json` 是人工扩展文件，只允许人工维护 `aliases` 和 `tags`；其它字段由 ingest/API 生成。
- `data/venue_aliases.json` 使用 `canonical / display / aliases`；匹配使用 canonical + aliases，展示使用 display。

### Citation Graph

- ingest 全量同步末尾生成 `data/paper_data/citation_graph.json`。
- 图只描述当前本地 active 论文之间的引用关系。
- 边方向固定为 `source -> target`：
  - `source`：引用发出论文
  - `target`：被引用的本地论文
- citation graph v1 匹配条件同时满足：
  - target canonical title 出现在 reference raw text 的 normalized 文本中
  - target 第一作者姓氏出现在 reference raw text token 中
  - reference raw text 中出现 target 年份候选之一
- 年份候选包括 `preprint_year`、`publish_year`、venue 字符串中的四位年份。
- `references.jsonl` 保留为原始引用证据；citation graph 是派生索引。

## 4. Retrieval Data/Common 边界

- `retrieval/data` 是本地数据执行层：读取 manifest/chunks/annotations/citation graph，做 record/filter/scope/chunk 级处理。
- `retrieval/domains/common` 是 parser/domain 共用基础设施：schema 校验、parser client、prompt/error，不读取本地数据。
- `data/utils.py` 中的 `normalize_bm25_token()` 和英文 `STOPWORDS` 保留：
  - 用于 BM25 英文 chunk
  - 用于英文 title/alias/manifest search
  - 用于翻译候选去重
  - 不作为中文问句的主解析工具
- 论文身份 key 全项目统一用 `manifest_records.paper_record_key()`。
- filter value 展平统一用 `data.utils.value_to_text_list()`。
- `parser_scope_resolver.py` 负责把 parser 输出中的 `paper` mention、`venue` alias、`year interval` 论文边界解析成规范值。
- `paper_scope_records.py` 负责 `semantic + filters + groups` 到候选论文 records。
- `filters.py` 只做单条 manifest record 的最终布尔匹配，不做 parser mention 解析。
- `aliases_match.py` 只做结构化 paper mention 的别名匹配，不改写用户 query。
- `chunks_load.py` 只负责 chunk 数据加载与按论文候选过滤。
- `citation_scope.py` 负责 `paper follow / paper prior` 这类基于 citation graph 的本地关系范围。

## 5. Parser Schema 与 Filter 规则

### Top

```json
{"router": "metadata|reference|content|unclear"}
```

- 只允许 `router` 字段。
- top parse 失败或 `unclear` 时，不进入三条 domain parser。

### 通用 Paper Filter

合法组合固定为：

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

`filters` 数组内多个条件默认是 AND。OR/PER/AND 分组通过 `paper_groups + group_mode` 表示，不用多个同字段 `=` filter 表达 OR。

### ArXiv Year 过滤

- `venue=ArXiv` 是特殊过滤语义，不要求 manifest 的 `venue` 写成 `ArXiv`。
- 只要论文有 `year.preprint_year`，就可被 `venue=ArXiv` 命中。
- 同一组 filters 中包含非否定 `venue=ArXiv` 时，`year` filter 使用 `preprint_year`。
- 普通 year filter 使用 `publish_year`。

## 6. Route 语义

### Metadata

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

- `lookup` 必须有 `return_fields`。
- `list` 没有 `return_fields` 时默认返回 `title`。
- `count/exists/null` 要求 `return_fields=[]`。
- `group_mode="and"` 只允许用于 `exists`。
- 执行层通过 `paper_scope_records.records_for_scope()` 查 manifest records。

### Reference

Reference 统一理解为：

```text
source_scope --cites--> object_scope
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

- `list/count` 要求 `return_side=source|object`。
- `exists/null` 要求 `return_side=null`。
- `return_side="source"` 返回引用发出方论文。
- `return_side="object"` 返回被引用方论文。
- 执行层优先使用本地 `citation_graph.json`；图缺失时返回 `status="graph_missing"` 和 warning，不临时扫描全库兜底。
- source/object 两侧的 filters 都先经过 `parser_scope_resolver` 标准化，再用 `paper_scope_records` 得到候选论文集合。

### Content

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

- `compare` 要求至少两个 `compare_objects`。
- 非 `compare` intent 要求 `compare_objects=[]`。
- `count/exists` 要求 `content_objects` 非空。
- `group_mode="and"` 只允许用于 `exists`。
- content 先用 `paper_scope_records` 限制候选论文，再只对命中论文的 chunks 做 dense/BM25。

### Content Retrieval Query

`domains/content/retrieval_query.py` 负责：

- `dense_query`：中文自然语言句子，服务 embedding，不拼接 paper/title/year/venue/author scope。
- `bm25_queries`：关键词候选列表，来源于：
  - `content_objects`
  - `compare_objects`
  - 从剩余 query 中抽出的核心词
  - 腾讯/阿里翻译候选
- 已结构化为 scope 的 paper/title/venue/year/author 值会从 query fallback 中扣掉，避免例如 `VIT` 同时作为论文范围和 BM25 关键词。
- `source_terms` 只在 debug 中用于解释 query 生成来源。

## 7. Evidence 输出

默认 composer 输出骨架：

```json
{
  "query": "...",
  "route": "metadata|reference|content",
  "status": "ok",
  "intent": "...",
  "plan": {},
  "resolved": {},
  "results": {},
  "warnings": []
}
```

压缩规则：

- 空数组、空对象、空字符串字段不输出。
- `resolved` 默认不输出；只有 alias 命中或必要消歧信息时输出简短 `aliases`。
- 完整 parser_result、RouteDecision、records、raw edges、context_units、retrieval source terms 只进 `debug`。
- `metadata`：
  - `lookup/list` 输出 `results.items`
  - `count` 输出 `results.count`
  - `exists` 输出 `results.exists`
- `reference`：
  - `list` 输出 `results.papers` 和精简 `results.edges`
  - `count` 输出 `results.count`
  - `exists` 输出 `results.exists`
  - edge 精简为 `source / object / ref / page / block`
- `content`：
  - 默认输出 `plan.retrieval_query.dense_query / bm25_queries`
  - 默认输出 `results.contexts`
  - 每个 context 只保留 `chunk_id / title / section_path / pages / text`
  - `expanded_blocks / sources / scores / scope_records` 只进 `debug`

`status` 只表示 planner 执行状态：

- `ok`
- `parse_failed`
- `graph_missing`
- `unclear`

无结果只写 warnings，不改变 `status`。

## 8. Dense / Sparse / Fusion

- `dense/service.py`：
  - `run_index()`：读取 chunks、请求 embedding、重建 Milvus collection
  - `run_search()`：CLI search 使用
  - `search_dense_chunks()`：content planner 使用
- `sparse/bm25.py`：
  - `BM25Index`：本地 BM25
  - `search_bm25_chunks()`：多个 BM25 query 分别检索，再用 RRF 合并
- `chunk_fusion.py`：
  - `RRF_K=60`
  - `fuse_chunk_hits()`：合并 dense 和 BM25 命中，按 `chunk_id` 去重

## 9. 配置

主要 `.env` 字段：

- MinerU：
  - `MINERU_API_KEY`
  - `MINERU_API_BASE_URL`
  - `MINERU_MODEL_VERSION`
  - `MINERU_LANGUAGE`
- 外部 metadata：
  - `DBLP_DELAY_SECONDS`
  - `DBLP_CANDIDATE_LIMIT`
  - `SEMANTIC_SCHOLAR_DELAY_SECONDS`
  - `SEMANTIC_SCHOLAR_API_KEY`
  - `ARXIV_DELAY_SECONDS`
- Plan parser：
  - `PLAN_PARSER_BASE_URL`
  - `PLAN_PARSER_API_KEY`
  - `PLAN_PARSER_MODEL`
  - `PLAN_PARSER_TIMEOUT_SECONDS`
- Content retrieval：
  - `PLAN_DENSE_TOP_K`
  - `PLAN_BM25_TOP_K`
  - `PLAN_FINAL_TOP_K`
  - `PLAN_BLOCK_WINDOW`
  - `PLAN_BM25_TRANSLATE_PROVIDERS`
  - `PLAN_BM25_TRANSLATE_TIMEOUT_SECONDS`
- Dense index：
  - `MILVUS_URI`
  - `MILVUS_TOKEN`
  - `MILVUS_DB_NAME`
  - `MILVUS_COLLECTION`
  - `EMBEDDING_BASE_URL`
  - `EMBEDDING_API_KEY`
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DIM`
  - `EMBEDDING_BATCH_SIZE`
  - `EMBEDDING_CACHE_PATH`
- BM25 keyword translation：
  - `TENCENT_TRANSLATE_SECRET_ID`
  - `TENCENT_TRANSLATE_SECRET_KEY`
  - `TENCENT_TRANSLATE_REGION`
  - `TENCENT_TRANSLATE_ENDPOINT`
  - `ALIYUN_TRANSLATE_ACCESS_KEY_ID`
  - `ALIYUN_TRANSLATE_ACCESS_KEY_SECRET`
  - `ALIYUN_TRANSLATE_REGION`
  - `ALIYUN_TRANSLATE_ENDPOINT`
  - `ALIYUN_TRANSLATE_VERSION`

密钥文件夹必须被 `.gitignore` 忽略，不进入 Git。

## 10. 测试与调试入口

常用测试：

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_content_route tests.test_bm25 -v
python -m compileall paper_rag
```

CLI smoke：

```powershell
python -m paper_rag --help
python -m paper_rag plan "BERT 是谁写的？"
python -m paper_rag plan "ResNet 的模型结构是什么？" --debug
```

Prompt / planner probe：

```powershell
python paper_rag/retrieval/domains/top/prompt_probe.py
python paper_rag/retrieval/domains/metadata/planner_probe.py --debug --show-route
python paper_rag/retrieval/domains/reference/planner_probe.py --debug --show-route
python paper_rag/retrieval/domains/content/planner_probe.py --debug --show-route
python paper_rag/retrieval/evidence_probe.py --route content --debug --show-route
```

## 11. 命名规范

- `validate_*`：schema / parser payload 校验，失败抛 `PlanParseError`。
- `normalize_*`：纯归一化，不读本地数据，不做检索。
- `resolve_*`：把 parser mention、别名、venue、年份边界解析成内部稳定值。
- `match_*`：布尔匹配。
- `filter_*`：集合过滤并返回子集。
- `search_*`：执行检索。
- `build_*`：组装结构化对象或 evidence。
- `to_evidence_*`：内部对象裁剪成对外 evidence 字段。
- `dedupe_*`：按明确 key 保序去重。

变量命名：

- `query`：用户原问题。
- `retrieval_query`：content 内部检索输入对象。
- `dense_query`：embedding 使用的中文自然语言检索句。
- `bm25_queries`：BM25 使用的关键词候选列表。
- `paper_semantic / filters / paper_groups / group_mode`：单侧论文范围结构。
- `source_* / object_*`：reference 两侧 scope。

## 12. 待接入

- `paper-rag ask` 和 answer composer。
- content context 到回答 LLM 的最终 prompt 组织。
- 腾讯/阿里翻译真实调用已接通，但仍需要在真实 plan 场景中继续观察 warning、配额和候选词质量。
