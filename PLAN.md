# Paper_RAG Maintenance Plan

## 1. 项目目标

把一组 PDF 论文整理成可检索、可追踪、可重建的本地知识库，并提供面向论文问答的 CLI/API 工作流

## 2. 当前共识

- 先构建项目框架，再执行具体功能实现
- 协作边界：除非用户明确要求拓展功能，否则不主动新增功能链路；默认只做问题分析、边界记录、bug 修正和已达成共识的实现。
- 论文原始文件统一放在 `data/pdf/`，当前只考虑 PDF 输入。
- 论文解析采用 MinerU 官方 API 实现。
- 唯一入口为 `paper-rag ingest`，每次运行执行一次全量同步。
- PDF 身份使用完整 SHA256 `file_hash`，避免重命名后被误判为未处理。
- PDF 的移动或重命名不会改变 `file_hash`；只要文件内容不变，就应复用同一份 MinerU 解析结果。
- 目录和文件名使用标题 slug、hash 短码或确认后的 `年份_标题`。
- 作者、年份和 venue 优先通过 DBLP 标题精确匹配自动确认；DBLP 未命中后查 Semantic Scholar，再未命中后查 ArXiv。
- Semantic Scholar 只作为 DBLP 后的补充召回源，命中后仍必须通过本地 normalized title 完全一致校验。
- ArXiv 命中时 `venue` 统一保存为 `ArXiv`。
- DBLP、Semantic Scholar 和 ArXiv 都未命中时，只提示 unresolved，不再通过 `metadata_overrides.json` 自动硬补。
- venue 命名暂时接受不同来源的原始形式；DBLP/Semantic Scholar venue 统一不是当前首要目标，后续如要处理再单独讨论 DBLP direct record 和 CoRR 降级边界。
- `metadata.json` 不保存 abstract 正文；abstract 作为独立论文内容区域参与后续普通索引。
- MinerU 解析结果至少保留四类结构化文件：
  - `content_list.json`：可读内容块按阅读顺序展平后的 flat list。
  - `content_list_v2.json`：按页组织，采用统一的 `type + content` 结构。
  - `layout.json`：中间结构化文件。
  - `model.json`：模型原始推理结果。
- 下游 metadata 构建以 `content_list_v2.json` 为主。
- 因为英文检索更好区分，所以先翻译再路由
- 代码里的注释用中文
- 写在PLAN.md里的数学公式以math形式写入
- router.py 产出完整的“计划路由决策对象”，包括 route / intent / return_field / filters / target_papers；planner.py 不再解析 metadata、不再决定二级语义，只按这个对象执行取证。
- 当前只有 metadata 链路完整可用；reference 和 content 先保留顶层 route 入口，reference 暂不做方向识别和 evidence 检索。

## 3. 架构草案

初步采用“原始文件 -> MinerU 解析结果 -> 内部稳定 schema -> chunk/index -> query”的分层架构。

关键原则：

- 原始 PDF、MinerU 原始输出、项目内部处理结果分层保存。
- 下游检索与生成不直接依赖 MinerU 原始 JSON，而是依赖项目自己的中间数据结构。
- `content_list_v2.json` 作为 metadata 构建的主要输入，但保留 `content_list.json`、`layout.json`、`model.json` 作为追踪、调试和回溯来源。

## 4. 模块划分

第一版围绕 `ingest` 建立以下模块，并按职责整理到 `paper_rag/` 包结构中：

- `paper_rag/cli/`：CLI 入口层，当前提供 `paper-rag ingest`、`paper-rag index`、`paper-rag search`、`paper-rag plan`、`paper-rag ask`；其中 `ask` 是薄入口，真正的答案组装逻辑放在 `retrieval/answer.py`。
- `paper_rag/dataprocess/`：PDF 同步、MinerU 调用、内容抽取和 paper_data 生成的数据处理层。
- `paper_rag/dataprocess/metadata/`：DBLP、Semantic Scholar、ArXiv 等外部论文元数据查询客户端。
- `paper_rag/retrieval/`：检索与证据规划层，按 `plan / dense / sparse / data` 分区保存。
- `paper_rag/config.py` 和 `paper_rag/utils.py`：跨模块基础设施，暂留根包。
- PDF 扫描与同步：扫描 `data/pdf/`，计算 hash，识别新增、删除和重复 PDF。
- Manifest 管理：维护 `data/manifest.jsonl`，记录 `file_hash`、当前 PDF 路径、title、status、MinerU 输出路径、paper_data 路径。
- manifest 中的 `message` 是 ingest 诊断字段，只用于记录错误、标题缺失、元数据未解析等状态说明；正常 active 记录可以为空或不写，不进入 `paper_data/metadata.json`。
- MinerU 客户端：封装 MinerU 官方 API 调用、任务轮询、结果下载和解压。
- DBLP 查询：用论文标题做 normalized exact match，命中后返回 author/year/venue。
- Semantic Scholar 查询：作为 DBLP 未命中后的补充召回，只接受 normalized title 完全一致；命中后返回 author/year/venue。
- ArXiv 查询：作为 Semantic Scholar 未命中后的兜底，只用标题精确匹配；命中后返回 author/year，`venue` 固定为 `ArXiv`。
- 内容抽取：从 `content_list_v2.json` 抽取标题、TOC、区域、正文 blocks、reference blocks。
- 数据落盘：维护 `data/mineru_output/`、`data/paper_data/` 和 `data/archive/`。

## 5. 目录结构

当前 `paper_rag/` 包按职责分层，后续新增模块优先落到明确职责目录，避免根目录和平铺文件继续膨胀：

```text
paper_rag/
  __init__.py
  __main__.py
  config.py
  utils.py

  cli/
    main.py
    ingest.py
    ask.py
    retrieval.py

  dataprocess/
    ingest.py
    manifest.py
    mineru.py
    extract.py
    metadata/
      dblp.py
      semantic_scholar.py
      arxiv.py
      retry.py

  retrieval/
    answer.py
    plan/
      planner.py
      router.py
      translation.py
      context.py
    dense/
      service.py
      embedding.py
      cache.py
      milvus_store.py
    sparse/
      bm25.py
    data/
      aliases.py
      chunks.py
      references.py
      manifest_lookup.py
```

目录职责：

- `cli/`：命令行入口和参数解析，只做薄封装，把实际工作委托给 dataprocess/retrieval 服务。
- `dataprocess/`：PDF 同步、MinerU 解析、论文内容抽取、manifest 和 paper_data 生成。
- `dataprocess/metadata/`：外部论文元数据查询客户端，不参与用户问题检索。
- `retrieval/plan/`：接入最终回答 LLM 前的证据链路；包含中文 query 翻译、英文规则顶层路由、各 route 的下层语义解析、evidence pack 编排和命中 block 扩展。
- `retrieval/dense/`：向量侧能力；包含 DashScope embedding、embedding cache、Milvus/Zilliz 存储和 `index/search` 服务。
- `retrieval/sparse/`：本地稀疏检索算法；第一版只包含 BM25 和 tokenizer。
- `retrieval/data/`：本地检索数据适配层；读取 `chunks.jsonl`、`references.jsonl` 和 `manifest.jsonl`。
- 当前不保留旧路径兼容包装；内部代码和测试统一使用新路径。
- 暂不新增未讨论清楚的 `ask/` 或 LLM 目录；等 ask 链路边界确定后再建立。

## 6. 数据流

`paper-rag ingest` 每次执行全量同步：

1. 扫描 `data/pdf/` 下所有 PDF。
2. 计算每个 PDF 的完整 SHA256 `file_hash`。
3. 先检查重复 hash：同 hash 多文件只保留一个处理对象，并在 CLI 汇总提示。
4. 根据 manifest 识别已删除 PDF：
   - MinerU 原始输出保留/归档到 `data/archive/`。
   - 对应 `data/paper_data/` 删除，因为它是可重建派生产物。
   - manifest 保留该 `file_hash` 的历史记录，并将状态标记为 deleted/archived。
5. 根据 manifest 识别新增 PDF。
6. 对新增 PDF 先按 `file_hash` 检查历史记录：
   - 如果 hash 对应的 MinerU 输出仍在 `data/mineru_output/`，直接复用。
   - 如果 hash 对应的 MinerU 输出已归档到 `data/archive/`，先恢复到 `data/mineru_output/`，不重新调用 MinerU。
   - 如果 hash 从未处理过，才调用 MinerU。
7. 从 MinerU 返回的 `content_list_v2.json` 中提取论文标题。
8. 生成标题 slug 和 8 位 hash 短码，派生数据落到 `data/paper_data/<title_slug>_<hash8>/`。
9. 使用标题查询 DBLP，要求 normalized title 精确匹配：
   - 命中：写入 author/year/venue，PDF 重命名为 `年份_标题.pdf`，MinerU 输出目录最终命名为 `年份_标题`。
   - DBLP 多个精确候选时，优先非 `CoRR` 记录；如果只有 `CoRR` 精确候选，也接受 `CoRR`。
   - 未命中或查询失败：继续查 Semantic Scholar。
   - Semantic Scholar 命中：写入 author/year/venue，PDF 重命名为 `年份_标题.pdf`，MinerU 输出目录最终命名为 `年份_标题`。
   - Semantic Scholar 未命中或查询失败：继续查 ArXiv。
   - ArXiv 命中：写入 author/year，并设置 `venue="ArXiv"`，PDF 重命名为 `年份_标题.pdf`，MinerU 输出目录最终命名为 `年份_标题`。
   - DBLP、Semantic Scholar 和 ArXiv 都未命中：保留解析结果，不重命名 PDF，CLI 汇总提示，退出码仍为 0。
10. 从 `content_list_v2.json` 构建 `metadata.json`、`toc.json`、`blocks.jsonl`、`references.jsonl`。

异常规则：

- 如果 MinerU 没有抽到任何 `title` block，CLI 明确提示具体 PDF 没有 title，不重命名。
- 如果最终 `年份_标题.pdf` 已存在且 hash 不同，报错并跳过该 PDF 的重命名，避免覆盖。
- 标题 slug 使用英文下划线风格：空格和标点转 `_`，重复 `_` 压缩。
- `data/paper_data/` 代表当前活跃检索库来源，只能包含 `data/pdf/` 中仍然存在的论文；PDF 删除后必须删除对应 paper_data，避免检索时命中已删除论文。
- 删除 PDF 后，如果同 hash 的 PDF 再次放回 `data/pdf/`，`ingest` 应从 `data/archive/` 恢复对应 MinerU 输出目录，不重新调用 MinerU，并重新生成对应 paper_data。
- 启动检索或问答前应先运行 `paper-rag ingest`，确保 `data/paper_data/` 已反映 PDF 的新增、删除和恢复。

## 7. 配置与环境

conda 下 RAG_project 环境运行

- 所有 API 密钥、服务 URL 和环境相关配置统一从项目根目录 `.env` 读取。
- `.env` 使用中文分区注释风格；后续新增配置必须放在对应分区下，或先新增清晰分区注释。
  - `# MINERU配置`
  - `# 检索配置`
  - `# 目录设置`
  - `# Chunk配置`
  - `# Milvus配置`
  - `# Embedding配置`
  - `# 翻译配置`
  - `# Plan配置`
  - `# Plan语义解析配置`
- `.env` 至少包含 MinerU API Key，例如 `MINERU_API_KEY`。
- MinerU API base URL 后续也通过 `.env` 配置，避免在代码中写死。
- MinerU 默认使用 `model_version=vlm`。
- MinerU 默认使用 `language=en`。
- DBLP/Semantic Scholar/ArXiv 查询默认一篇篇顺序执行，并在查询之间保持延迟，避免触发限流。
- DBLP/Semantic Scholar/ArXiv 的网络请求对临时失败重试一次：timeout、临时 `URLError`、HTTP `429/500/502/503/504`；明确失败如 `403/404` 不重试，正常返回但标题未命中也不重试。
- `DBLP_DELAY_SECONDS` 控制连续 DBLP 查询之间的间隔，默认 1 秒。
- `DBLP_CANDIDATE_LIMIT` 控制 DBLP publication search 返回候选数，默认 20。
- `SEMANTIC_SCHOLAR_DELAY_SECONDS` 控制连续 Semantic Scholar 查询之间的间隔，默认 5 秒；Semantic Scholar 只在 DBLP 未命中后查询。
- `SEMANTIC_SCHOLAR_API_KEY` 为可选配置，存在时通过 `x-api-key` 请求头访问 Semantic Scholar；为空时使用公开限流接口。
- `ARXIV_DELAY_SECONDS` 控制连续 ArXiv 查询之间的间隔，默认 3 秒；ArXiv 只在 Semantic Scholar 未命中后查询。
- 输入目录通过 `.env` 的 `PDF_DIR` 配置，默认 `data/pdf`。
- MinerU 原始输出目录通过 `.env` 的 `MINERU_DIR` 配置，默认 `data/mineru_output`。
- 论文派生数据目录通过 `.env` 的 `PAPER_DIR` 配置，默认 `data/paper_data`。
- chunk 目标长度通过 `.env` 的 `CHUNK_TARGET_CHARS` 配置，默认 1400。
- chunk overlap 通过 `.env` 的 `CHUNK_OVERLAP_CHARS` 配置，默认 200。
- Milvus/Zilliz 通过 `.env` 配置：
  - `MILVUS_URI`
  - `MILVUS_TOKEN`
  - `MILVUS_DB_NAME`
  - `MILVUS_COLLECTION=paper_rag_chunks`
- Embedding 通过 DashScope OpenAI 兼容接口配置：
  - `EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `EMBEDDING_API_KEY`
  - `EMBEDDING_MODEL=text-embedding-v4`
  - `EMBEDDING_DIM=1024`
  - `EMBEDDING_BATCH_SIZE=10`
  - `EMBEDDING_CACHE_PATH=data/index/embedding_cache.jsonl`
- 百度翻译通过 `.env` 配置，用于将中文问题转成英文检索 query：
  - `BAIDU_TRANSLATE_APP_ID`
  - `BAIDU_TRANSLATE_SECRET_KEY`
  - `BAIDU_TRANSLATE_ENDPOINT=https://fanyi-api.baidu.com/api/trans/vip/fieldtranslate`
  - `BAIDU_TRANSLATE_DOMAIN=academic`
  - 第一版使用百度领域文本翻译的 academic 领域；签名按领域翻译规则拼接 `appid + q + salt + domain + secret_key`。
- `paper-rag plan` 证据链路参数通过 `.env` 配置：
  - `PLAN_DENSE_TOP_K=20`
  - `PLAN_BM25_TOP_K=20`
  - `PLAN_FINAL_TOP_K=8`
  - `PLAN_BLOCK_WINDOW=2`
- plan 语义解析通过独立 OpenAI-compatible parser 配置；该 parser 用于 `metadata / reference / content` 三个顶层 route 的第二层结构化解析，不生成最终答案：
  - `PLAN_PARSER_BASE_URL`
  - `PLAN_PARSER_API_KEY`
  - `PLAN_PARSER_MODEL`
  - `PLAN_PARSER_TIMEOUT_SECONDS=30`
- 删除归档目录：`data/archive/`。
- `data/paper_data/` 是后续检索库的数据源；它不是长期归档目录，而是由当前活跃 PDF 集合和 MinerU 解析结果重建出来的活跃数据层。

CLI 行为：

- `paper-rag ingest`：执行全量同步；已有 author/year/venue 时不查外部元数据源，缺失时尝试 DBLP -> Semantic Scholar -> ArXiv。
- 默认 `ingest` 复用已有完整 metadata 时，应同时复用 manifest 中的规范 title，避免被 MinerU 原始标题大小写覆盖。
- `paper-rag ingest --refresh`：强制忽略 manifest 中已有元数据，对全库重新执行 DBLP -> Semantic Scholar -> ArXiv 元数据刷新。
- `paper-rag index`：消费现有 `data/paper_data/*/chunks.jsonl`，调用 embedding 并重建 Milvus collection；不会自动运行 `ingest`，避免意外触发 MinerU 或外部元数据请求。
- `paper-rag search "query" --top-k 5`：对 query 做 embedding，在 Milvus 中进行 chunk 级向量召回，输出 score、title、section、pages、chunk_id 和 snippet。
- `paper-rag plan "问题"`：生成接入最终回答 LLM 前的 JSON evidence pack；只做路由、结构化解析、检索和证据收集，不生成最终自然语言答案。各顶层 route 的第二层语义解析可以调用独立 plan parser LLM。
- `paper-rag ask "问题"`：当前已打通 metadata v1，基于 `plan` 的 evidence pack 用确定性模板生成可读答案；reference/content 仍只保留入口，后续再补。

## 8. 检索策略

检索层负责把用户问题转成结构化 evidence pack，不生成最终自然语言答案。

### 8.1 入口与通用规则

- `paper-rag plan "问题"` 是接入最终回答 LLM 前的证据规划与证据收集层；默认输出 JSON evidence pack。
- 用户问题预计主要是中文；检测到中文时先调用百度领域翻译 API 得到英文 `retrieval_query`，英文输入则不翻译。
- 主路径固定为“先翻译，再路由”：规则路由只读取英文 `retrieval_query`；`original_query` 只用于追踪展示。
- 中文翻译失败时返回 `route=error` 和空 evidence，在 `warnings` 中写入 `translation_failed`，CLI 进程仍正常退出。
- 顶层路由为 `metadata / reference / content`，优先级为 `reference > metadata > content`。
- 第一层路由不调用 LLM，使用英文 token + 高置信短语/句型规则；命中顶层 route 后的第二层语义解析可调用独立 plan parser。
- evidence pack 顶层字段固定包含 `original_query / retrieval_query / route / intent / return_field / router_reason / evidence / warnings`；不再使用旧的 `sub_route` 字段，也不预留 `reference_direction` 字段。
- 不保留顶层 `language` 字段；后续检索统一消费英文 `retrieval_query`。
- evidence 不使用 `scope` 和 `expanded_query`；范围限定统一由 `target_papers` 表达，内部 alias/canonical 扩展不暴露。
- 第二层语义解析是 `metadata / reference / content` 三个分支共用的设计原则，但每个分支的下沉 schema 可以不同：
  - metadata 重点解析字段查询、论文列表、数量统计、字段过滤和否定过滤。
  - reference 重点解析 `cite / cited` 引用方向、目标论文、引用范围过滤和引用列表/统计意图。
  - content 重点解析正文问题类型、目标论文范围、比较对象和需要召回的内容焦点。
  - 当前先细化 metadata schema；reference/content 的 parser schema 后续单独讨论，避免把不同任务硬塞进同一个结构。

### 8.2 Metadata Route

- metadata route 不进入 Milvus，不检索 chunks；只读取 `data/manifest.jsonl` 中的 active 记录。

- 第一层 metadata 入口采用保守规则：只有英文 `retrieval_query` 命中明显作者、年份、venue、标题、论文列表或统计线索时才进入 metadata route；其他非 reference 问题直接落到 content。

- metadata 命中后，不再用硬规则细分 `paper_list / author / year / venue / title`；改为调用 OpenAI-compatible plan parser 的 metadata 解析提示词，输出严格 JSON。

- plan parser 在 metadata 分支只做结构化解析，不回答问题，不生成自然语言解释；prompt 必须明确要求只输出 JSON。

- metadata 分支 parser 输出 schema：

  ```json
  {
  "router": "metadata",
  "intent": "lookup",
  "return_field": "author",
  "filters": [],
  "raw_query": ""
  }
  ```

- parser 枚举约束：
  - `intent` 只能是 `lookup / list / count / unknown`。
  - `return_field` 只能是 `author / year / venue / title / null`；只有 `lookup` 时最关键。
  - `filters[].field` 只能是 `author / year / venue / title`。
  - `filters[].op` 只能是 `= / in / contains / between`。
  - `filters[].value` 可以是字符串、数字、字符串列表或区间列表，例如 `"Kaiming He"`、`2015`、`["ResNet", "Transformer"]`、`[2015, 2020]`。
  - `filters[].negated` 必须显式输出 `true / false`，不能只靠自然语言表达否定。
  - 不允许输出 `!=`、`not contains`、`not in` 等非 schema 运算符；否定只能通过 `negated=true` 表达，且 op 仍保持正向。
  - `not on ArXiv` / “不在 ArXiv 上发布”必须解析为 `field=venue / op=contains / value=ArXiv / negated=true`，不能写成 `not contains`。
  - `raw_query` 保存翻译后的英文 query；如果原问题本来是英文，则保存原英文 query。
  - 不确定时使用 `unknown` 或 `null`，不要猜测字段或意图。
  
- parser 失败边界：
  - HTTP 超时或错误、非 JSON、非法枚举、缺少必需字段时，metadata evidence 标记 `parse_failed`，写入 warning，不回退 content，也不使用旧硬规则猜答案。
  - `intent=unknown` 时不执行 manifest 查询，只返回 parser 结果和 warning。
  
- metadata evidence 行为：
  - `lookup`：需要 `return_field`；按 title/alias/entity/filter 定位论文，再返回目标字段值。
  - `list`：按 filters 查询 `manifest.jsonl`，返回匹配论文列表。
  - `count`：按 filters 查询 `manifest.jsonl`，同时返回 `count` 和匹配论文列表，保证可追溯。
  - 作者过滤必须完整匹配作者名，不支持 `He` 这种姓氏短匹配直接命中 `Kaiming He`。
  - 否定过滤通过 `negated=true` 表达，例如“不在 ArXiv 上发布”解析为 `venue contains ArXiv` 且 `negated=true`。
  
- metadata evidence 返回匹配论文的 `title / author / year / venue / pdf_path / paper_data_path` 等字段；plan 只返回证据，不直接生成自然语言答案。

### 8.3 Reference Route

- reference route 当前只保留顶层入口，不进入 Milvus，不查本地 `references.jsonl`，不做 cite/cited 方向识别。
- 进入条件为英文 query 命中完整 token：`reference / references / referenced / referencing / bibliography / bibliographies / citation / citations / cite / cites / cited / citing`。
- reference 命中时 evidence 返回 `parse_status=not_implemented` 和空 `references`，并在 warnings 中提示 reference evidence 尚未实现。
- ask v1 不对 reference 生成最终答案；只要 route 进入 reference，就统一返回明确的未实现提示，避免把未完成的引用链路伪装成可答状态。
- reference 命中后的第二层也属于 plan parser 的职责，后续应使用 reference 专用提示词解析引用方向、目标论文、列表/统计意图和范围过滤；reference schema 单独讨论后再实现。
- 在 reference schema 确定前，不输出 `reference_direction`，不做 reference intent cleanup，不运行 reference BM25。

### 8.4 Content Route

- content route 是默认兜底；当前不做正文子路由，`intent=None`。
- content 命中后的第二层也属于 plan parser 的职责，后续应使用 content 专用提示词解析正文问题类型、比较对象、目标论文范围、问题焦点和需要召回的内容范围。
- content parser 的子路由暂定继续围绕 `lookup / reasoning / comparison / synthesis` 讨论，但本阶段不实现硬规则 content 子路由。
- ask v1 不对 content 直接生成自然语言答案；content 先只通过 `plan` 产出 evidence pack，后续等正文回答层设计稳定后再接入生成。
- content route 使用 `abstract + body` 的 `chunks.jsonl` 作为召回输入；不使用 reference，不默认使用 appendix。
- Milvus dense 召回 top `PLAN_DENSE_TOP_K` chunks，默认 20。
- 本地 BM25 在 chunks 上召回 top `PLAN_BM25_TOP_K` chunks，默认 20。
- 如果 `target_papers` 非空，只检索这些目标论文的 chunks；如果为空，则全库检索。
- 使用 RRF 融合 dense 和 BM25 排名，按 `chunk_id` 去重，最终取 top `PLAN_FINAL_TOP_K`，默认 8。
- 命中 chunk 后用 `block_ids` 回查 `blocks.jsonl`，并按同 section 前后 `PLAN_BLOCK_WINDOW` 个 blocks 扩展，默认 2。
- content evidence 输出 `context_units`，每个 unit 包含 chunk 来源、融合分数、dense/BM25 来源、section_path、pages、chunk_text、expanded_blocks。
- `blocks.jsonl` 用于精读、上下文扩展、图片/表格/公式还原和引用定位；`chunks.jsonl` 只作为召回输入。

### 8.5 索引与 BM25

- `paper-rag index` 消费现有 `data/paper_data/*/chunks.jsonl`，不会自动运行 `ingest`。
- `paper-rag index` 每次删除并重建 Milvus collection，避免 PDF 删除、chunk 变化或维度变化造成脏索引。
- embedding 使用 DashScope OpenAI 兼容接口，默认模型 `text-embedding-v4`，默认维度 1024；客户端使用标准库 HTTP，不引入 `openai` SDK。
- embedding cache 按 `model + dim + embedding_text` 的 SHA256 缓存向量；chunk 文本、模型或维度变化会自然 cache miss。
- Milvus 记录除 vector 外保存展示字段：`chunk_id`、`paper_id`、`chunk_index`、`title`、`section_path_text`、`pages_text`、`text`。
- `paper-rag search` 第一版直接返回 Milvus chunk 展示字段，不回查 blocks；block 扩展留给 `plan/ask` 链路。
- BM25 第一版本地轻量实现，不新增依赖；content 粒度是 chunks，reference 粒度是 reference items。
- BM25 tokenizer 做规范化：统一小写，标准化 Unicode 连字符和下划线，保留数字、英文缩写、连字符词和公式相关 token。
- BM25 停用词采用保守英文小表；参数固定为 `k1=1.5`、`b=0.75`。
- BM25 评分公式：

  $$
  \operatorname{score}(q, d)
  =
  \sum_{t \in q}
  \operatorname{IDF}(t)
  \cdot
  \frac{f(t,d)\,(k_1 + 1)}
  {f(t,d) + k_1 \left(1 - b + b \cdot \frac{|d|}{\operatorname{avgdl}}\right)}
  $$

  $$
  \operatorname{IDF}(t)
  =
  \log\left(1 + \frac{N - \operatorname{df}(t) + 0.5}{\operatorname{df}(t) + 0.5}\right)
  $$

- 其中 `f(t,d)` 是 token 在文档中的词频，`|d|` 是过滤停用词后的文档 token 数，`avgdl` 是平均文档长度，`N` 是文档总数，`df(t)` 是包含该 token 的文档数。

## 9. 生成策略

生成层第一版只打通 metadata 问答，`paper-rag ask` 作为 plan evidence 的可读答案出口。

- `paper-rag ask "问题"` 复用 `paper-rag plan` 的 evidence pack；当前只对 metadata 用确定性模板输出自然语言答案。
- ask 不重新做 PDF ingest、MinerU 解析或外部 metadata 查询；它只消费当前 evidence pack 和本地 `paper_data`。
- metadata evidence 用于回答作者、年份、venue、标题和论文列表类问题；lookup/list/count 都可直接出答案。
- reference 和 content 仍保留入口，但 ask v1 只对 metadata 生成答案，reference/content 先统一返回明确的未实现提示。
- 生成回答时保留简短 provenance；metadata 结果仍可追溯到 manifest/paper_data。
- 第一版 ask 不在本阶段实现 rerank、长答案规划、引用网络图、DOI/DBLP 解析或联网补全。

## 10. 抽取与 Chunk 规则

普通 RAG chunking 的默认输入范围：

- 使用 `abstract + body`。
- 不使用 `reference`。
- `appendix` 写入 `blocks.jsonl` 并标记为 `region=appendix`，但默认不进入普通索引。

block 处理规则：

- `title`：进入 TOC 和正文上下文。
- `paragraph`：进入正文 block。
- `list`：普通列表进入正文 block；`reference_list` 进入 `references.jsonl`。
- `equation_interline`：进入正文 block。
- `image`：保存相对 MinerU 输出目录的 `source_path`，保存 `caption`，`text=caption`，不做多模态理解。
- `table`：保存相对 MinerU 输出目录的 `source_path`、`caption` 和 `html`，`text` 使用确定性的半结构化文本供后续索引，不做 AI 总结。
  - 如果第一行像表头，转为 `Columns: col1, col2...`。
  - 后续每行转为 `Row n: col1 = value1; col2 = value2...`。
  - 如果没有明确表头，使用 `column_1 / column_2 / ...`。
- `image_footnote` / `table_footnote` 第一版暂不保存。
- `page_footnote`：第一版完全忽略。
- `page_aside_text`：像页眉、页脚一样默认忽略。
- 页码、页眉、页脚默认不进入普通索引。

标题抽取规则：

- 先定位 Abstract marker：`type=title` 且文本归一化后为 `Abstract`，或第一个以 `Abstract.` / `Abstract:` 开头的 paragraph。
- 优先取 Abstract marker 前的非特殊 `title` block 作为论文标题。
- 特殊标题包括 `Abstract`、`References`、`Appendix`、`Acknowledgements` 等不应作为论文标题的结构性标题。
- 判断特殊标题前先剥数字章节编号，例如 `6 Acknowledgement`、`7 References`、`3 Appendix` 分别按 `Acknowledgement`、`References`、`Appendix` 判断。
- 如果 Abstract 前没有可用标题，则 fallback 到第一页的 `page_header`。
- 不使用 PDF 文件名兜底，避免 `2604.pdf` 这类文件名污染元数据。
- 外部元数据源精确命中后，metadata title 使用外部规范标题，用于修正 MinerU OCR、空格、公式下标和大小写问题。
- 如果仍无法获得标题，CLI 明确提示具体 PDF 没有 title，不重命名。

区域划分规则：

- `abstract`：优先从 `type=title` 且文本为 `Abstract` 的 block 后开始，到下一个正文一级章节标题前结束。
- 如果没有找到 `Abstract` title，则 fallback 到第一个以 `Abstract.` 或 `Abstract:` 开头的 paragraph，并从该 paragraph 中去掉 `Abstract.` / `Abstract:` 前缀后抽取 abstract 内容。
- `Keywords` / `Index Terms` / `CCS Concepts` 属于前置信息，不进入 `abstract` 或 `body` 的普通检索 blocks。
- 如果没有找到任何 Abstract marker，标记抽取 warning 并提示用户，但不中断整批 ingest。
- `body`：正文一级章节开始后到 References 前。
- `reference`：References 后的参考文献区域。
- `appendix`：References 后如果再次出现新的一级标题，则从该标题开始判断为附录，并结束 reference 区域；不要求标题必须叫 Appendix。
- 如果 References 前出现标题 `Appendix`，则从该标题到 References 前也归为 `appendix`，不进入普通索引。
- `Acknowledgements` / `Funding` / `Disclosure` 及其内容默认不进入普通索引。
- 编号形式的致谢/资金/披露标题也按特殊标题处理，例如 `6 Acknowledgement`、`8 ACKNOWLEDGMENTS` 不进入 TOC 或 `blocks.jsonl`。
- 暂不新增前置信息/脚注硬过滤规则，例如 `Equal contribution`、`Correspondence to`、`Proceedings of` 第一版先保留。
- 论文标题到 Abstract 之前的前置内容只用于标题、作者、年份等 metadata 构建，后续不再作为普通检索内容使用。

TOC 构建规则：

- 优先用标题编号推断层级，例如 `3`、`3.2`、`3.2.1`。
- MinerU 的 `level` 只作备用，因为示例中所有标题可能都被标为 level 1。
- TOC 需要服务后续按 section 聚合 chunk，因此应保留章节树形结构和章节顺序。
- TOC 只覆盖 `abstract + body`，不为 `reference` 和 `appendix` 建树。
- `Abstract` 作为特殊 section 保存，`section_id` 为 `sec_abstract`。
- 如果标题具有明确数字编号，则按编号关系展开为树：
  - `1`
  - `2`
  - `3`
  - `3.1`
  - `3.1.1`
  - `3.1.2`
- 如果全文正文标题没有明确数字编号体系，则按正常章节顺序保存为同级顶层章节索引。
- 在已经存在明确数字章节体系的论文中，无编号 `title` 默认不新建顶层 section；如果它位于正文区域且不是关键词/致谢类特殊标题，则作为当前 section 下的 inline heading 写入 `blocks.jsonl`。
- `Broader Impact` / `Limitations` 等有内容价值的无编号标题按 inline heading 保留。
- `section_id` 采用编号优先 slug：
  - 有编号标题：`3.2.1 Scaled Dot-Product Attention` -> `sec_3_2_1`。
  - 无编号标题：`Abstract` -> `sec_abstract`。

内部数据文件结构：

- `metadata.json` 只保存简洁论文级信息：`title`、`author`、`year`、`venue`、`pdf_path`。
- `toc.json` 同时保存树形结构和扁平 section 索引：
  - `sections`：扁平列表，用于按 `section_id` 聚合 chunk；字段包含 `section_id`、`title`、`number`、`level`、`parent_id`、`path`、`start_block_index`、`end_block_index`、`region`。
  - `tree`：树形结构，用于展示和结构化导航；无明确编号的论文退化为同级顶层列表。
- `blocks.jsonl` 每行保存一个可检索内容 block，最小字段为：
  - `block_id`
  - `order`
  - `region`
  - `type`
  - `text`
  - `page`
  - `bbox`
  - `section_id`
  - `section_path`
- `references.jsonl` 在 `ingest` 阶段按条拆分参考文献，但暂不联网解析 DBLP/DOI；每行最小字段为：
  - `reference_id`
  - `ref_index`
  - `raw_text`
  - `page`
  - `source_block_id`
- `chunks.jsonl` 是后续检索输入层，由 `ingest` 在生成 `blocks.jsonl` 后同步生成；默认只覆盖 `abstract + body`，不生成 `appendix/reference` chunks。
- chunk 采用 section 内聚合策略：
  - 按 `section_id` / `section_path` 分组。
  - 同一 section 内按 block `order` 聚合。
  - 默认目标长度 1400 字符，超过后按 block 顺序切为下一 chunk。
  - 单个 block 超过目标长度时独立保留，不强拆长表格或长段落。
  - section 内多 chunk 时，`embedding_text` 拼接上一 chunk 尾部最多 200 字符作为 overlap；`text` 保持干净主体，overlap 不计入精确引用来源。
- `chunks.jsonl` 每行最小字段为：
  - `chunk_id`
  - `paper_id`
  - `chunk_index`
  - `region`
  - `section_id`
  - `section_path`
  - `pages`
  - `block_ids`
  - `text`
  - `embedding_text`
  - `char_count`
- `chunk_id` 使用 `<paper_data_dir.name>::chunk_0000` 形式，`chunk_index` 与 `chunk_id` 后缀都从 0 开始，`paper_id` 使用 `paper_data_dir.name`。
- `chunk_id` 和 `chunk_index` 暂时都保留：
  - `chunk_id` 是全库唯一字符串 ID，后续可直接作为向量库主键或检索结果引用 ID。
  - `chunk_index` 是论文内部的 0-based 顺序号，便于排序、相邻 chunk 查找和重建论文内顺序。
  - 两者必须保持一致：`chunk_id` 后缀 `chunk_0000` 对应 `chunk_index=0`。
- `text` 保存干净的 chunk 主体文本，不加入论文标题或章节前缀；`embedding_text` 用于后续 embedding，包含 `Paper: <title>` 和 `Section: <section_path>` 短前缀。
- chunk 中 table 使用半结构化文本，image 使用 caption。
- `equation_interline` 在 chunk 中作为公式类型文本处理：
  - 短公式以 `Equation: ...` 形式加入 chunk，例如 `Equation: L = ...`。
  - `Equation:` 只是确定性的类型标签，用来告诉 embedding 和后续调试“这段是公式”，不是把公式翻译成自然语言，也不做 AI 总结。
  - 公式语义主要依赖前后 paragraph，公式文本只用于召回损失函数、符号、模型结构等精确信息。
  - 过长公式第一版不进入 chunk，避免大段符号污染语义检索；是否需要公式专门索引后续再讨论。
- `reference_list` 必须限定在 `reference` 区域内才拆入 `references.jsonl`；正文区域中被 MinerU 误标为 `reference_list` 的普通列表不能进入参考文献链路。
- reference item 没有 item 级 bbox 时不保留 `bbox` 字段。
- reference 编号规则：
  - 如果原文有 `[21]` 这类显式编号，优先解析并保存为 `ref_index=21`。
  - 如果没有显式编号，则按 reference 区域中的出现顺序从 1 递增。
  - 原始编号形式不单独保存为 `label`；如需查看，保留在 `raw_text` 中。

## 11. 评估与测试

第一版测试重点：

- 用 Attention Is All You Need 的 `content_list_v2.json` 验证标题抽取、编号 TOC、abstract/body/reference 区域切分。
- 验证 Center Loss 的 `Abstract.` paragraph fallback 会去掉前缀，且 `Keywords` 不进入普通 blocks。
- 验证 Center Loss、LSTM、NormFace 这类编号致谢标题不会进入 TOC/body blocks。
- 验证 EfficientNet 可从第一页 `page_header` fallback 得到标题。
- 验证 Inception-v4 这类无编号章节体系生成同级 TOC。
- 验证 BERT、ResNet、SENet 这类 References 后 appendix 能正确截断 reference，并以 `region=appendix` 写入 blocks。
- 验证图片/表格 block 保留结构字段：image 的 `source_path/caption`，table 的 `source_path/caption/html`，且表格 `text` 转为半结构化 `Columns` / `Row` 文本。
- 验证 `chunks.jsonl` 自动生成，且只包含 `abstract/body`。
- 验证 chunk schema 包含 `paper_id`、`section_path`、`pages`、`block_ids`、`text`、`embedding_text`。
- 验证 `embedding_text` 有 Paper/Section 前缀和 overlap，`text` 不包含前缀且不混入 overlap。
- 验证长 section 会切成多个 chunks，单个超长 block 不被硬拆。
- 验证 `chunk_id` 后缀和 `chunk_index` 都从 0 开始且保持一致。
- 验证短公式进入 chunk 时带 `Equation:` 标签，长公式不强行进入 chunk。
- 验证 `.env.example` 包含 Milvus 与 Embedding 分区，`Settings` 能读取 URI、token、collection、model、dim、batch 和 cache path。
- 验证 DashScope/OpenAI-compatible embedding 请求：
  - URL 为 `<EMBEDDING_BASE_URL>/embeddings`。
  - Header 使用 `Authorization: Bearer <EMBEDDING_API_KEY>`。
  - Payload 包含 `model`、`input`、`dimensions=1024`。
- 验证 embedding cache 命中时不发 HTTP 请求；文本、模型或维度变化时重新请求。
- 验证 `paper-rag index` 只读取 chunks，不触发 ingest/MinerU/metadata，并会重建 Milvus collection。
- 验证 `paper-rag search "query" --top-k 5` 返回 chunk 级结果，包含 score、title、section、pages、chunk_id 和 snippet。
- 验证 `paper-rag plan "..."` 输出合法 JSON evidence pack，不生成最终自然语言答案；各顶层 route 的第二层语义解析允许调用独立 plan parser LLM。
- 验证 plan router：
  - `Who wrote / Who are the authors` 进入 metadata route，并由 parser 输出 `intent=lookup / return_field=author`。
  - `When was published / publication year / what year` 进入 metadata route，并由 parser 输出 `intent=lookup / return_field=year`。
  - `Which journal / Which conference / venue` 进入 metadata route，并由 parser 输出 `intent=lookup / return_field=venue`。
  - `What is the title / title of` 进入 metadata route，并由 parser 输出 `intent=lookup / return_field=title`。
- `Who are the authors of ResNet and Transformer respectively` 进入 metadata route，并由 parser 输出 `intent=lookup / return_field=author`，多个目标自动分组。
  - `Which papers were published in 2015-2019` 进入 metadata route，并由 parser 输出 `intent=list` 和 `year between [2015, 2019]` filter。
  - `Which papers are written by Kaiming He` 进入 metadata route，并由 parser 输出 `intent=list` 和 `author = Kaiming He` filter；作者完整姓名匹配，`He` 不应匹配 `Kaiming He`。
  - `How many papers were published in 2019` 进入 metadata route，并由 parser 输出 `intent=count` 和 `year = 2019` filter。
  - `Which papers were not published on ArXiv between 2015 and 2020` 进入 metadata route，并由 parser 输出 `year between [2015, 2020]` 以及 `venue contains ArXiv` 且 `negated=true`。
  - 英文 reference、referenced、citation、bibliography、cited、cite 类完整 token 问题进入 reference route。
  - reference/citation/bibliography/cite/cited 等完整 token 问题进入 reference route，但当前只返回 `parse_status=not_implemented`，不做引用方向识别和 reference evidence 检索。
  - `recite` 不误命中 reference，`authorization` 不误命中 metadata。
  - 其他英文问题默认进入 `content` route，`intent=None`。
  - 中文 metadata/reference/body query 必须先经百度翻译得到英文 `retrieval_query`，再触发路由。
- 验证百度翻译：
  - 中文 query 调百度翻译得到英文 `retrieval_query`。
  - 英文 query 不调用翻译。
  - 翻译失败时写入 warning，返回 `route=error`，不回退原 query 检索。
- 验证 metadata evidence：
  - 从 `data/manifest.jsonl` 读取 active 记录。
  - parser 返回非法 JSON、非法枚举、缺必需字段、HTTP 超时或错误时，metadata evidence 标记 `parse_failed` 并写入 warning，不落到 content。
  - parser 返回 `intent=unknown` 时不查询 manifest，只返回 parser 结果和 warning。
  - `lookup` 按 target/entity/title alias 匹配 manifest，并返回目标字段值。
  - `list` 按 filters 查询 manifest 并返回匹配论文列表。
  - `count` 按 filters 查询 manifest，同时返回 `count` 和匹配论文列表。
  - 多目标 lookup 时 evidence 按目标论文或实体分组返回。
  - evidence 返回 metadata 字段，但 plan 不直接回答。
- 验证 reference route 入口：
  - reference/citation/bibliography/cite/cited 等完整 token 问题进入 reference route。
  - evidence 返回 `parse_status=not_implemented` 和空 `references`，不运行 BM25，不联网解析 DOI/DBLP。
- 验证 body route：
  - mock Milvus dense top20 + 本地 BM25 top20。
  - 使用 RRF 融合，按 `chunk_id` 去重，输出 top8。
  - 命中 chunk 能回查 `blocks.jsonl`，并按同 section 前后 2 blocks 扩展成 `context_unit`。
- 验证 DBLP 精确标题命中后得到年份、作者和 venue，并触发 PDF/MinerU 输出最终命名。
- 验证 DBLP 未命中时继续查 Semantic Scholar；Semantic Scholar 命中时必须通过 normalized title 完全一致校验。
- 验证 Semantic Scholar 未命中时继续查 ArXiv；ArXiv 命中时 `venue="ArXiv"`。
- 验证 DBLP、Semantic Scholar 和 ArXiv 都未命中时退出码为 0、状态为 unresolved、PDF 不重命名。
- 验证重复 hash PDF 不重复调用 MinerU。
- 验证删除 PDF 后 MinerU 数据归档、`paper_data` 删除、manifest 状态更新。
- 验证 `reference_list` 在 ingest 阶段拆成条目级 `references.jsonl`，但不参与普通 chunking。

## 12. 待决策问题

- reference 专门链路后续如何基于 `references.jsonl` 解析 DBLP/DOI/标题/作者/年份/venue。
- `ask` 如何消费 `plan` evidence pack、如何组织 prompt、如何生成引用。
- body route 后续是否需要细分 method/experiment/result/definition/comparison 等子路由。
- 是否需要 reference 编号定位，例如 `[12]` 或“第 12 篇引用”。

