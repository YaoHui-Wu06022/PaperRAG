# LlamaIndex 索引与 Embedding 增量同步

实现位于 `llamaindex/nodes.py`、`embedding.py`、`index.py` 和 `service.py`。本文记录当前代码的执行方式，特别区分全量 staging 切换与增量原地更新；二者的失败恢复能力不同。

## 1. 原文、Catalog 与向量分开更新

```text
ArXiv 下载 → MinerU 解析 → catalog sync → index rebuild
```

下载/解析不自动同步 Catalog；Catalog 同步不调用 DashScope 或 Milvus；索引更新才加载有效正文节点、请求 Embedding 并写 Milvus。仅修改 Query Rewriter、翻译、JEV、RRF 或 Agent 提示词，不改变既有 Chunk，通常不需要重建向量。

LlamaIndex 直接使用已有 Chunk，不重新切块。TextNode.id_ 是 chunk_id；text 是 retrieval_text，metadata.content_text 保存展示原文。元数据包含论文、章节、区域、类型、页码、来源 Block、资源和哈希。所有 metadata 键从 Embedding 文本拼接中排除，避免重复加入身份信息。

索引包含 abstract/content/appendix，不含 reference。默认查询只搜 abstract/content，并不表示 appendix 未被向量化。

## 2. Embedding 客户端

DashScope 客户端 POST `<DASHSCOPE_BASE_URL>/embeddings`，使用 EMBEDDING_MODEL、EMBEDDING_DIMENSIONS、encoding_format=float。索引输入是 retrieval_text；查询输入是原始用户问题。

默认维度 1024，批次默认 20，配置值限制到 1–20。客户端按响应 index 排序并校验返回数量、每个向量的长度，再转 float。超时和重试复用 JsonHttpClient；LlamaIndex BaseEmbedding 适配器提供同步单条、同步批量和异步查询接口，异步查询目前调用同步方法，并非真正非阻塞 HTTP。

查询 Embedding 与文档 Embedding 缓存不是同一件事：一次正文请求共用 QueryBundle.embedding，comparison 各篇复用；不同请求的 query 向量没有持久化缓存。文档缓存定义如下。

## 3. 缓存键及保存位置

```text
chunk_id
retrieval_text_hash = sha256(retrieval_text.encode("utf-8"))
embedding_model
embedding_dimensions
chunk_rule_version
```

SQLite embedding_items 保存上述字段以及 milvus_collection、synced_at，不存向量本体；向量仍在 Milvus。embedding_state 是 key/value 表，记录 active_collection、catalog_indexed_at、模型/维度/规则、last_sync_*、last_reused/added/updated/deleted/failed 和 stale_reasons。

Catalog 重建尝试复制上一份数据库的缓存表。缺列或不可读取时不兼容转换，记录 cache_schema_mismatch，下一次 auto 转全量。日常新增/删除导致缓存暂时覆盖不全是正常增量输入，不因此必然全量重建。

Chunk ID 使用版本、整份 content_list 文件哈希、规则版本、ordinal 和源文本派生。因此重新解析导致文件哈希或顺序变化时，即使某段正文相同，其 chunk_id 也可能变化，统计会表现为 added+deleted，而不是 updated。规则升级会令全部 ID 改变，缓存节省不是跨任意解析版本的内容去重。

## 4. 模式判定

| 请求 | 实际选择 |
|---|---|
| full | 所有当前 Chunk 重新 Embedding |
| auto | 契约边界不兼容时 full，否则 incremental |
| incremental | 边界不兼容时返回 rebuild_required，不静默全量 |

强制全量的边界原因：manifest_missing、manifest_schema_mismatch、chunk_rule_version_mismatch、embedding_model_mismatch、embedding_dimensions_mismatch、collection_missing、cache_missing、cache_schema_mismatch。

catalog_timestamp_mismatch、chunk_count_mismatch、cache_incomplete 等内容更新原因允许增量。Manifest schema 当前版本为 2；Chunk 规则为 content-list-regions-v6-boundary-aware。没有可索引正文节点时失败，不创建空集合替换现有索引。

## 5. 同步统计

| 字段 | 定义 |
|---|---|
| reused | 缓存键相同且增量处理中继续使用的 Chunk |
| added | 当前 ID 不在缓存，或增量实查发现向量缺失需补写 |
| updated | 同 ID 的缓存键变化；全量时旧缓存中仍存在的当前 ID |
| deleted | 缓存或实际集合中存在、当前 Catalog 已没有的 ID |
| failed | 当前批次 Embedding/写入失败时计入的 Chunk 数；不是所有验证失败的统一逐条统计 |

full 的 reused 为 0，旧缓存未包含的全部当前 ID 计 added。增量会强一致查询集合 ID，修正缺失与多余记录，不能只看 SQLite 缓存认为向量一定存在。

## 6. 全量：临时 Collection 与 Manifest 切换

1. 从 SQLite 加载节点，分类并保存旧缓存快照。
2. 创建 `<collection>__build_<随机标识>` staging Collection。
3. 对新增/更新节点分批请求 Embedding，向 staging 写入带预计算向量的 TextNode。
4. 核对集合主键集合与 Catalog ID 集合完全相同，再抽查一个节点向量维度。
5. SQLite 事务替换 embedding_items/state，写入本次统计。
6. 先写临时 Manifest，再原子替换 manifest.json。
7. 清理旧集合，清除服务缓存，后续加载新 active。

验证使用强一致主键查询，不只依赖可能延迟的 row_count；“探测”是按 chunk_id 查询一条向量，不是检索质量测试，也没有逐条验证每项 metadata 语义。

正常的写入、验证或 Manifest 写入失败发生在切换完成之前时，清理 staging、恢复 SQLite 缓存，旧 Manifest 和旧 active 保留。Ctrl+C 也有 staging 清理分支。旧集合清理异常被忽略，可能留下需检查的旧集合；不能把文件替换称为跨 SQLite、文件系统和 Milvus 的分布式事务。

## 7. 增量：当前是 active Collection 原地更新

实际代码走 `_rebuild_active_collection()`，**没有创建新的临时 Collection，也不复制全部未变向量**：

1. 强一致读取 active 的主键，识别 missing/stale ID。
2. 只对 added/updated 请求 Embedding，以 upsert_mode 写入 active。
3. 删除 deleted 节点并 flush。
4. 验证主键集合和一条向量维度。
5. 更新 SQLite 缓存与 Manifest，集合名称保持不变。

完全不变的 auto 更新仍会连接 Milvus、核对 ID、验证和更新 Manifest/状态，但 changed_ids 为空时不会请求文档 Embedding，也没有向量写入。

这里的失败恢复只恢复缓存表，**不回滚已经成功执行的 Milvus upsert/delete**。因此不能承诺增量失败时旧 active 内容完全不变，也不能承诺读请求在增量期间只看到更新前快照。重试增量或执行 full 是修复持久化差异的手段；本次文档更新不改该实现。

## 8. Manifest 和状态的含义

默认 Manifest 路径 `data/index/llamaindex/manifest.json`，保存 schema_version、catalog_indexed_at、chunk_count、indexed_count、chunk_rule_version、embedding_model、embedding_dimensions、milvus_collection、built_at、status、last_sync_mode、sync_stats。文件 JSON 明确使用 UTF-8。

`library_index_status` 外层 status 当前为 ok；真正就绪状态在 `data.status` 和 `data.index_ready`，不能只看外层。data.index_stale 与 stale_reasons 给出：

```text
manifest_missing / manifest_schema_mismatch
catalog_timestamp_mismatch / chunk_count_mismatch
chunk_rule_version_mismatch
embedding_model_mismatch / embedding_dimensions_mismatch
collection_missing / cache_missing / cache_incomplete / cache_schema_mismatch
last_sync_failed
```

cache_complete 主要判断缓存 ID 集合与 Catalog 相同且 collection 字段一致，不逐项重新计算 retrieval_text_hash，也不检查所有实际向量。collection_missing 可能是集合真不存在，也可能是连接失败被存在检查转为 false。

load 首次读取时拒绝 stale 索引，正文可降级词法；同一 IndexService 已加载后直接返回内存缓存，不在每次 query 再执行全部 stale 检查。服务层按 Settings 缓存最多四个 IndexService，成功 rebuild 后清理缓存。外部进程更新 Catalog/Manifest 时，长期运行 MCP 不保证即时重新加载，宜重启服务或通过当前服务完成管理流程。

## 9. 命令与确认边界

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag catalog sync --json
conda run -n RAG_project python -X utf8 -m paper_rag index status --json
conda run -n RAG_project python -X utf8 -m paper_rag index rebuild --mode auto --json
conda run -n RAG_project python -X utf8 -m paper_rag index rebuild --mode incremental --json
conda run -n RAG_project python -X utf8 -m paper_rag index rebuild --mode full --json
```

这些 rebuild 命令会写数据；CLI 直接同步执行，不要求 confirm。MCP 则先预览，再 `library_index_rebuild(confirm=true, mode="auto")` 排队执行，读取 job.result 内状态。不要把三种模式示例当作每次都必须连续执行的流程。

修改图查询或提示词不应自动同步 Catalog/向量。修改表格检索文本、公式上下文、章节切分或模型/维度时，先完成 Chunk 检查，再按上述契约选择同步。真实召回记录与自动测试见 [record.md](record.md) 和 tests/test_llamaindex_service.py。
