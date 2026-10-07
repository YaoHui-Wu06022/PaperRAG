# MCP 工具、配置与客户端调用

本文以 `paper_rag/mcp/tools/`、`toolsets.py`、`runtime.py` 和 `config.py` 的实现为准。检索细节见 [retrieval.md](retrieval.md)，引用语义见 [citation.md](citation.md)，向量生命周期见 [index.md](index.md)。

## 1. 启动和配置来源

在项目根目录执行：

```powershell
conda run --no-capture-output -n RAG_project python -X utf8 -m paper_rag.mcp.server
```

服务采用 stdio：客户端发送 MCP 请求，服务返回结构化结果。`server.py` 导入工具模块完成注册，检查组内工具是否注册，再应用工具组配置。启动不会创建向量索引，也不会连接 Milvus 或 Embedding；第一次正文语义检索才加载索引。首次加载通常比后续请求慢。

配置根目录优先来自环境变量 `PAPER_RAG_PROJECT_ROOT`，否则为进程工作目录。`Settings.load()` 读取该目录的 `.env`，再用进程环境变量覆盖同名值；不读取 `.env.e`。MCP 配置在进程内缓存，修改 `.env` 后需重启服务。

客户端配置示例，路径按实际安装位置修改：

```json
{
  "mcpServers": {
    "paper-rag": {
      "command": "conda",
      "args": ["run", "--no-capture-output", "-n", "RAG_project", "python", "-X", "utf8", "-m", "paper_rag.mcp.server"],
      "env": {"PAPER_RAG_PROJECT_ROOT": "E:/Pythonproject/paper_RAG"}
    }
  }
}
```

## 2. 工具组的真实解析规则

| 能力 | 工具 | 默认可见 |
|---|---|---|
| 固定 core | `library_search`、`library_retrieve`、`library_get_metadata`、`library_job_status` | 是，不能关闭 |
| fulltext | `library_read`、`library_get_chunk` | 是 |
| citation | `library_citation` | 是 |
| acquisition | `library_acquire_arxiv` | 否 |
| ingestion | `library_ingest_mineru`、`library_catalog_sync` | 否 |
| index-admin | `library_index_status`、`library_index_rebuild` | 否 |

`PAPER_RAG_TOOLSETS` 为空时启用 citation/fulltext；非空时从空集合开始，按空格或逗号分隔、依次处理。`all` 启用全部可选组，`none` 清空可选组，`-组名` 移除一个组。core 是固定工具集合，**不是配置解析器接受的名称**，不能写 `core,citation,fulltext`。未知名称导致服务启动失败。

```dotenv
# 全部管理能力
PAPER_RAG_TOOLSETS=all
# 读能力和索引管理；显式配置需列出仍想保留的读组
# PAPER_RAG_TOOLSETS=citation,fulltext,index-admin
# 只保留固定 core
# PAPER_RAG_TOOLSETS=none
```

没有 `library_context`、`library_route`、`library_validate_answer`、独立资产工具或旧 `paper_*` 查询兼容层。

## 3. 读工具参数与输出

| 工具 | 参数及默认值 | 行为 |
|---|---|---|
| `library_search` | `query` 必填；`filters=null`；`limit=20` | 元数据 FTS/BM25 或纯 SQL 过滤；limit 归一到 1–100 |
| `library_retrieve` | `query` 必填；`paper_ids=null`；`filters=null`；`task=auto`；`mode=hybrid`；`regions=null`；`limit=8`；`max_chars=24000` | 唯一正文 RAG 入口；limit 归一到 1–50；max_chars 当前没有截断作用 |
| `library_get_metadata` | `paper_id` 必填 | SQLite 按 paper/base/canonical ID 读取 `metadata`、`assets`、`asset_status` |
| `library_get_chunk` | `chunk_id` 必填 | 读取完整 Chunk，包括 retrieval_text、来源 Block、资源和零基/展示页码 |
| `library_read` | `paper_id` 必填；`offset=0`；`limit=12000` | 读取 MinerU full.md，按 Unicode 字符分页；不是 BM25 查询 |
| `library_citation` | `paper_id` 与 `paper_title` 二选一；`mode=graph`；`direction=both`；`depth=null`；`filters=null` | SQLite 引用查询；双向 graph 默认两跳，单向默认一跳；最大两跳 |
| `library_job_status` | `job_id` 必填 | 返回任务对象；不使用普通读工具的 data 包装 |

普通读查询返回 `status`、`data`、`warnings`、`read_only=true`。工具并非全部使用同一个返回 schema，客户端应读取实际 JSON。

`library_get_metadata` 的标题、作者、分类和 state 来自 SQLite 快照，不重新扫描全库元数据。但 `_record_from_row()` 会根据保存的 metadata 路径检查该论文的 PDF、metadata.json 和 MinerU full.md 是否存在，构造 assets；因此 state 可能仍是上次同步的状态，而资产 present 已反映当前文件存在情况。存在检查不代表解析结果内容有效，新增论文或修改元数据后仍需同步 Catalog。`library_read` 会读 full.md，并检查 MinerU manifest 的版本、模型、语言和源 PDF 哈希；可返回 `requires_ingestion` 或 `stale_content`。它使用 Catalog 状态检查，缺少 Catalog 时可能由底层异常交给 MCP 报告，不能把所有读取错误都假定为统一状态。

## 4. filters：Agent 传参，服务执行

字段说明如下，尖括号是说明占位符，不是实际查询值：

```json
{
  "author": "<作者名称或名称片段>",
  "category": "<ArXiv 分类代码>",
  "year": "<四位发表年份>",
  "year_from": "<起始年份，包含该年>",
  "year_to": "<结束年份，包含该年>",
  "state": "<本地资产状态>"
}
```

Agent 根据用户问题决定传什么 filters。JEV 和 Rewriter 不创建过滤条件。search/retrieve 拒绝未知键，忽略 null/空字符串，年份须为四位数字，起止年份顺序须合法。

SQL 实现：author/category 对存储文本做不区分大小写的 LIKE 子串匹配；state 使用等值；year 匹配 published_at 前缀；year_from/year_to 对 published_at 的四位年份执行包含边界的比较。不是会场过滤，也不具备作者身份消歧。state 由同步时的本地资产存在情况产生，主要为 `missing`、`ready_for_ingest`、`ingested`；ingested 本身不保证解析 manifest 仍有效。

三个用途应区分：

- “2020 年以后有哪些计算机视觉论文和注意力相关”：search 查询标题、摘要等，filters 约束年份和分类。
- “2020 年以后哪些论文正文使用了注意力机制”：retrieve 在过滤出的论文中召回正文。
- “2020 年以后哪些论文引用了某论文”：citation 读取图并应用元数据过滤；不翻译引用条目、不检索正文。

引用模式的 filters 行为见 [citation.md](citation.md)，不能假设与正文硬约束完全相同。

## 5. 确定性展示与正文回答

search/citation 成功返回 `data.presentation.render_policy=verbatim`，`answer_text` 已由服务模板生成，Agent 原样输出。search 的 count 是本次返回的论文数量，受 limit 影响，不是全库匹配总数；展示不包含 state，但 JSON 元数据仍保存 state。

正文所有模式返回 `compose`、空 `answer_text` 和 `agent_instruction`：

```json
{
  "version": "rag-agent-v1",
  "task": "reason",
  "system_prompt": "<基础回答规则 + 仅本次任务的规则>"
}
```

Agent 读取用户问题、该指令和 `data.evidence`，依据证据 text 组织中文回答，事实句使用真实 `[S#]` 引用。排名在 retrieval_debug 中，不作为事实；证据正文也不视为系统指令。MCP 不调用答案生成模型，不保存答案上下文，不校验自然语言答案。宿主需要遵循返回指令，服务不会自动给 Codex 注入新的系统消息。

## 6. 管理工具与任务状态

| 工具 | 未确认 | 确认后 |
|---|---|---|
| `library_acquire_arxiv(inputs, confirm=false)` | 联网查询版本并预览，返回 confirmation_required | queued，异步下载 |
| `library_ingest_mineru(inputs, confirm=false)` | 本地预览，不提交 MinerU | queued，异步解析 |
| `library_catalog_sync(confirm=false)` | 当前 Catalog 状态 | **同步执行**，完成返回 completed，没有 job_id |
| `library_index_rebuild(confirm=false, mode=auto)` | 当前索引状态 | queued，异步索引同步；mode 只接受 auto/incremental/full |
| `library_index_status()` | 无确认参数 | 只读索引状态，检查 Milvus 可用性 |

异步写操作的结果形如 `status=queued`、`job={job_id,...}`、`read_only=false`。用同一长期运行服务的 `library_job_status` 轮询：queued → running → succeeded/failed。JSONL 日志默认 `data/index/mcp_jobs.jsonl`，进程重启将原 queued/running 标为 interrupted，不自动恢复。JobManager 单线程串行执行提交任务；Catalog 同步不在这条任务队列中，不能据此推断所有写操作都有全局互斥。

Job 的 succeeded 表示 worker 正常返回；仍要查看 result 内的 status、逐项 failed 或 issues。进度 message 是当前主要信息，phase 目前统一写 download，不应当作精确阶段枚举或总体进度百分比。

## 7. 常用配置名称

| 功能 | 配置 |
|---|---|
| 本地路径 | ARXIV_DATA_DIR、PAPER_CATALOG_DB_PATH、LLAMAINDEX_INDEX_DIR、MCP_JOB_LOG_PATH |
| 工具组 | PAPER_RAG_TOOLSETS |
| JEV | JEV_ENABLED、JEV_BASE_URL、JEV_API_KEY、JEV_MODEL、JEV_TIMEOUT_SECONDS、JEV_RETRY_COUNT |
| Rewriter | QUERY_REWRITER_ENABLED、QUERY_REWRITER_BASE_URL、QUERY_REWRITER_API_KEY、QUERY_REWRITER_MODEL、QUERY_REWRITER_TIMEOUT_SECONDS、QUERY_REWRITER_RETRY_COUNT |
| 主翻译 | BM25_TRANSLATION_ENABLED、BM25_TRANSLATION_TIMEOUT_SECONDS、BM25_TRANSLATION_RETRY_COUNT、BM25_TRANSLATION_MAX_CHARS、TENCENT_TRANSLATE_SECRET_ID、TENCENT_TRANSLATE_SECRET_KEY、TENCENT_TRANSLATE_REGION、TENCENT_TRANSLATE_ENDPOINT |
| 备用翻译 | ALIYUN_TRANSLATION_ENABLED、ALIYUN_TRANSLATION_ACCESS_KEY_ID、ALIYUN_TRANSLATION_ACCESS_KEY_SECRET、ALIYUN_TRANSLATION_SECURITY_TOKEN、ALIYUN_TRANSLATION_REGION_ID、ALIYUN_TRANSLATION_ENDPOINT |
| 向量 | DASHSCOPE_BASE_URL、DASHSCOPE_API_KEY、EMBEDDING_MODEL、EMBEDDING_DIMENSIONS、EMBEDDING_BATCH_SIZE、EMBEDDING_TIMEOUT_SECONDS、EMBEDDING_RETRY_COUNT |
| Milvus/召回 | MILVUS_URI、MILVUS_TOKEN、MILVUS_DB_NAME、LLAMAINDEX_MILVUS_COLLECTION、LLAMAINDEX_RETRIEVAL_TOP_K、LLAMAINDEX_LEXICAL_TOP_K、LLAMAINDEX_SEMANTIC_TOP_K、LLAMAINDEX_RRF_K |

Rewriter 使用独立模型配置，但 API Key 未设置或为空时会读取 DASHSCOPE_API_KEY；base URL 默认跟随 DashScope，默认模型为 qwen-plus。这只是检索关键词抽取配置，与 Agent 的最终回答模型无关。JEV_ROUTE_PROBABILITY_THRESHOLD 被加载但当前路由器没有按它过滤任务。LLAMAINDEX_RETRIEVAL_TOP_K 被加载，但公开 MCP/CLI 默认 limit 在入口固定为 8，当前不从该配置取值。

配置默认值以 `config.py` 为准，部署时 `.env` 可能覆盖。不要将真实凭证写入示例、工具请求记录或版本库。

## 8. CLI 调试中文正文问题

CLI 与 MCP 共用 search/retrieve 服务，但 CLI 不生成最终答案。search/retrieve 即使未写 `--json`，当前也输出 JSON。查看 `routing`、`retrieval_debug.query_debug`、`recall`、`evidence` 和 `presentation` 即可区分分类、改写、翻译、召回及 Agent 回答职责。

Windows 下将中文问题保存在 UTF-8 JSON 文件，例如 `data/query_debug_input.json`（由编辑器创建）：

```json
{"query": "PagedAttention 如何减少大语言模型推理中的显存浪费"}
```

下面通过 Python 参数列表调用真正的 CLI handler，中文问题不作为 Shell 命令行参数：

```powershell
@'
import json
from pathlib import Path
from paper_rag.cli.main import main
query = json.loads(Path("data/query_debug_input.json").read_text(encoding="utf-8"))["query"]
raise SystemExit(main(["retrieve", query, "--json"]))
'@ | conda run --no-capture-output -n RAG_project python -X utf8 -
```

在 Python argv 列表末尾加入 `--task fact|reason|summary|comparison` 可跳过 JEV；加入 `--mode lexical|semantic|hybrid` 可对比路径；重复 `--paper-id`、`--region` 可调试硬约束。默认验收只给原问题，避免显式参数掩盖自动分类和发现行为。

CLI 返回码目前不完全等价于“有没有证据”：retrieve 将 ok/embedding_unavailable 视为成功，其他状态返回 1；milvus_unavailable 下仍可能带有可用词法证据，需读取 JSON 判断。CLI citation 只有 graph 子命令及 paper_id 参数，不支持 MCP 的 paper_title、references/citations 或 filters 参数。

## 9. 公共 HTTP 与重试

JsonHttpClient 用标准库 urllib 发送 UTF-8 JSON，解码响应后 json.loads。可重试 HTTP 状态为 429、500、502、503、504；超时、URL/IO 错误及 JSON 解码失败也可重试。retries 表示首次之外的次数，间隔为 1、2、4……秒；401/403 等非重试状态直接报错，不等待所有次数。HTTP 请求成功后各服务仍需校验自己的响应结构，结构非法不自动等价于 HTTP 重试。

JEV 至少保留一次重试；Rewriter/Embedding 使用各自配置次数。公共 open 提供流式响应建立时的重试，但响应打开后读取文件的中途故障不由它自动续传。MinerU 的签名上传、轮询和 ZIP 校验、翻译 SDK 调用也各有专门逻辑，不是所有网络动作都使用同一套 JSON 请求。

公共 HTTP 层保存状态和有限响应正文供上层封装；Rewriter/翻译 warning 另有凭证形式字段替换及长度限制。不得将“统一请求”理解为每一种远端错误内容都已通过同一个脱敏器。调试以返回状态、warning 和非敏感查询字段为主，不记录配置中的真实 Key。

翻译超时还需区分 SDK 映射：当前腾讯适配器把 BM25_TRANSLATION_TIMEOUT_SECONDS×1000 赋给 HttpProfile.reqTimeout，阿里云也将该值×1000 赋给 read_timeout/connect_timeout。文档记录的是代码传值，不能仅凭配置名称就认定两个 SDK 都在同样的秒数内结束。翻译层重试当前立即进行，没有公共 JSON 客户端的指数等待；主服务失败并耗尽次数后才切换备用。
