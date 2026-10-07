# 正文检索实现与指标

实际 MCP 请求、响应与 Codex 回答见 [record.md](record.md)。本页说明 Query Rewriter、候选范围、混合召回和四类证据组织策略。SQLite 是论文及 Chunk 的读取来源，Milvus 保存既有向量；这些检索改动不要求同步 Catalog 或重建向量。

## 工具与执行顺序

Agent 选择顶层工具；JEV 不参与工具选择。

| 工具 | 职责 | 最终回答 |
|---|---|---|
| library_search | 标题、摘要、作者、分类等元数据发现 | 原样输出 presentation.answer_text |
| library_citation | SQLite 本地引用关系 | 原样输出 presentation.answer_text |
| library_retrieve | 正文证据召回 | 根据 evidence 与 agent_instruction 组织中文答案 |

正文流程：

```text
校验输入和 Catalog
→ JEV（仅 task=auto）或显式 task
→ 一次 Query Rewrite 和翻译
→ 硬约束、明确对象解析
→ 必要时混合发现目标论文
→ 正文召回与任务证据组织
→ 最终 limit、source_id、presentation
```

默认 mode=hybrid、limit=8。两路各自使用 LLAMAINDEX_LEXICAL_TOP_K、LLAMAINDEX_SEMANTIC_TOP_K，当前默认均为 50；最终 limit 不提前截断候选池。max_chars 参数目前没有用于截断证据，本次保持原状。

## Query Rewriter：对象与检索目标分开

严格返回两个字符串数组：

```json
{"entities": ["LoRA", "QLoRA"], "core_terms": ["显存占用"]}
```

entities 是原问题明确提到的论文、方法或模型名称，不能被当作已经确认的论文 ID。core_terms 是主题、具体属性、机制或比较维度。允许一个数组为空，不能同时为空；拒绝旧单字段、phrases、额外字段、非数组、非字符串与空字符串。同一名称同时出现时优先保留为 entity。

### 元数据用途

library_search 指定 purpose=metadata，不调用 JEV。元数据与正文使用同一版基础提示词，抽取原句中的命名对象与最小充分的检索主题。已经由 filters 表达的年份、作者、分类、状态不重复提取；Rewriter 不生成或修改 filters。

```text
问题：2020 年以后有哪些计算机视觉论文和注意力相关
Agent 参数：filters.year_from=2020，filters.category=cs.CV
预期改写：entities=[]，core_terms=["注意力"]
翻译：attention
FTS："attention"
```

纯过滤发现使用 query=""，直接查询 SQLite，不调用改写或翻译。

### 正文用途

library_retrieve 在任务确定后指定 purpose=body。fact、reason、summary、comparison 与元数据共用同一版基础提示词，不再拼接用途或任务提示词。purpose 和 task 保留为调用上下文及调试信息；任务仍由 JEV 或显式参数确定，用于后续证据组织。

名称和完整短语不做同义词扩展，不猜测论文全称；问题文本不视为系统指令。
泛指的“论文、文章”不属于方法名称；解析标题时允许去掉这类尾部措辞，原始 entities 仍完整保留在调试字段中。

### 翻译和阶段查询

原始提取与译文分别保留。仅中文项调用翻译，已有英文名称与缩写保持原样；腾讯云优先，失败时尝试已配置的阿里云。

元数据发现、全库正文召回使用 entities 与 core_terms 的完整短语并集，构造安全的 FTS5 OR 表达式。比较目标确定后，逐篇 BM25 优先使用比较维度，例如 "memory usage"；维度没有词法命中时，仅在同篇论文内回退对象名称。这个回退不再次调用模型或翻译。

Rewriter 超时或非法响应：原问题 → 翻译 → 停用词清理 → BM25。回退普通词不能被当作已经确认的实体。语义检索始终使用原始问题。

## 硬约束与目标发现

paper_ids 与 filters 同时存在时取交集。支持 author、category、year、year_from、year_to、state。显式 ID 不存在或被 filters 排除时返回 not_found，不静默忽略其中一篇。

默认区域为 abstract/content。只有问题明确出现附录、appendix 等，或指定 regions 时纳入 appendix。reference 始终禁止。区域和 ID 约束在 SQLite LIMIT 前、Milvus MetadataFilters、reason 扩展和章节补充阶段均生效。

fact/reason 没有 ID 时直接在允许范围内做 BM25 与向量召回，元数据或标题命中不再成为向量过滤条件。元数据 FTS 空结果不会阻断 hybrid。

summary/comparison 可以解析明确对象：

1. 完整标题规范化后精确匹配。
2. 方法名在标题中独立且唯一匹配。
3. 多个标题命中记录为 ambiguous，不取词法第一名直接锁定。
4. 摘要、正文提及不能确认论文身份。

目标不明确时，从允许范围做混合发现；按每篇论文首次出现在融合结果中的顺序，summary 选一篇，comparison 选最相关两篇。已经明确解析出的比较对象优先保留。显式 paper_ids 指定多篇时全部保留；comparison 不足两篇或目标数超过 limit 时返回 invalid_input。

## 两路召回与排序

SQLite chunks_fts 索引 retrieval_text，使用 FTS5/BM25 排序，包含完整章节前缀和表格纯文本。Milvus 使用原始问题向量及相同的区域、论文元数据过滤。

同一次请求共用 QueryBundle；首次语义检索生成的问题向量由后续发现和逐篇检索复用。Embedding 失败后不在每篇再次尝试。

按 chunk_id 去重并计算：

```text
RRF = 1 / (rrf_k + lexical_rank) + 1 / (rrf_k + semantic_rank)
```

缺少某路排名时不计该项，默认 rrf_k=60。普通问题以 RRF 为主要排序，实体、章节、区域及类型特征仅用于同分细分。score 为 RRF 分数，不代表事实正确概率。

表格、公式与文本统一参加排序，不整体后置、不给额外类型降权。明确表号、图号匹配优先于一般相关性；问题明确指向某篇论文时，唯一标题匹配仅用于编号查询的排序优先，不改变全库召回范围。普通问题仍保留图片、图表的后置处理。表格原始 HTML、检索纯文本和公式 LaTeX 不变。

## 四类任务如何选择最终证据

### fact

按融合顺序取直接证据，按 Chunk ID 去重，最多 limit 条。不扩展邻域，也不做章节补充。

### reason

遍历完整候选池，取消前一半核心限制。同论文、同区域、相同完整 section_path 内扩展相邻 ordinal，必要时修复半词边界。重叠窗口合并，完全覆盖的核心不再重复输出；继续选择后续候选直到 limit 或候选耗尽。

合并按原文顺序进行，并去掉原切块的重叠文本；同步合并页码、资源、来源 Block 和 source_chunk_ids。最终窗口沿用最高排名核心的顺序。count 是窗口数，不是原始 Chunk 数，四个窗口可能覆盖更多 Chunk。

### summary

每篇目标论文依次覆盖：摘要 → 方法 → 实验/结果 → 结论/讨论 → 引言及其他章节。章节类别根据现有完整路径及固定中英文关键词确定，不调用新模型。

每类优先取融合池最高排名 Chunk；该类无召回时按 ordinal 从对应章节补首个有效 Chunk，记录 chapter_supplements，不伪造排名。未知标题按顶层章节轮流补充；已覆盖后再放其余相关来源。致谢不参与补充，不存在的章节不补写。

多篇目标先轮流分配证据名额，避免仅覆盖第一篇。

### comparison

每篇分别召回、分别计算论文内 RRF，复用一次查询改写、翻译和问题向量。优先使用比较维度匹配的正文；配额至少两条时保留一条摘要背景。维度词法为空时，仅在同篇用对象名回退。

平均分配名额，余数按目标顺序分配，某篇不足的名额轮流补给其他论文。缺少有效 Chunk 的论文返回 no_evidence_for_paper warning，不能用另一篇替代结论。按论文轮流输出证据，最后统一分配 source_id。

## 返回证据与调试

data.evidence 保留现有字段：source_id、paper_id、chunk_id、text、type、section_path、page_start、page_end、score；reason 另含 source_chunk_ids。页码沿用解析器的零基编号。

改写调试位于元数据 data.query_debug 或正文 data.retrieval_debug.query_debug：

```json
{
  "purpose": "body",
  "task": "comparison",
  "entities": ["LoRA", "QLoRA"],
  "core_terms": ["显存占用"],
  "translated_entities": ["LoRA", "QLoRA"],
  "translated_core_terms": ["memory usage"],
  "lexical_query": "\"lora\" OR \"qlora\" OR \"memory usage\"",
  "rewriter_used": true,
  "rewriter_fallback": false,
  "translation_used": true,
  "translation_provider": "tencent",
  "translation_fallback": false
}
```

lexical_query 在 query_debug 中描述准备好的全局查询；每篇 comparison 实际使用的维度查询、对象回退查询见 recall.<paper_id>。

每阶段 recall.<阶段>.chunk_rankings 保留任务组织前的 Chunk 排名：lexical_rank 为 BM25 排名，semantic_rank 为向量排名，retrieval_rank 为融合排序后的顺序；缺少某一路命中时该路为 null。semantic_score 和 rrf_score 同处调试区。reason 合并窗口可通过 evidence.source_chunk_ids 对照原始 Chunk 的排名；这些详细排名不放入 data.evidence，不作为答案事实依据。

retrieval_debug 同时记录 constraint_paper_ids、target_paper_ids、entity_matches、preferred_paper_ids、recall、chapter_supplements、reason_windows、final_evidence_count、per_paper_evidence_count。preferred_paper_ids 是编号查询的软排序线索，不是 Milvus 过滤。recall 的每阶段包含 lexical_count、semantic_count、fused_count，便于区分底层无召回与组织后合并。

## Agent 答案组织

`library_retrieve` 是 Agent 的证据工具，不在服务内调用答案生成模型，也不保存答案上下文。正文响应的 `data.presentation` 还会返回结构化 `agent_instruction`：

```json
{
  "version": "rag-agent-v1",
  "task": "reason",
  "system_prompt": "你是论文库客户端回答 Agent……"
}
```

`agent_instruction` 只包含本次任务的回答规则，不复制正文。中文输出、`[S#]` 引用格式和 `data.evidence` 作为唯一事实来源的约束都写在 `system_prompt` 中。返回的 `evidence` 只包含检索到的 Chunk 正文、来源定位和评分；Agent 应以 `evidence[*].text` 为事实依据。

Agent 读取 `data.presentation.render_policy`：`verbatim` 且 `answer_text` 非空时原样输出；`compose` 时读取 `agent_instruction.system_prompt`、`data.query` 和 `data.evidence`，将证据按 `source_id` 组织后生成中文答案。`retrieval_debug`、内部排序字段和工具调用过程不进入事实证据区。

四类正文任务的规则如下：

- `fact`：直接回答问题，合并支持同一事实的多个 Chunk；
- `reason`：解释机制或因果关系，区分证据明确说明的内容和推断；
- `summary`：按问题、方法、实验结果和结论综合组织；
- `comparison`：按共同维度比较各论文或方法，并分别引用双方证据。

`insufficient_evidence` 响应仍返回 `agent_instruction`，但 `data.evidence` 为空。Agent 只能输出“当前检索结果没有包含足够的正文证据，无法可靠回答该问题。”，不能生成事实性答案或不存在的 `[S#]` 引用。

工具边界如下：

```text
Agent -> library_search / library_citation -> MCP 确定性 answer_text -> Agent 原样输出
Agent -> library_retrieve -> MCP agent_instruction + Chunk evidence -> Agent 组织答案
```

`library_search` 和 `library_citation` 返回已经组织好的 `data.presentation.answer_text`，Agent 直接原样输出；正文问题由 Agent 根据 `library_retrieve` 的多条证据自行组织答案。

## 失败回退和可观测性

| 情况 | 行为 |
|---|---|
| Rewriter 超时、非法双字段、两个数组都空 | 原问题翻译和停用词清理；query_rewriter_failed warning |
| 腾讯云翻译失败 | 尝试阿里云；记录原提供商失败及备用切换 |
| 所有翻译失败 | 改写成功时保留原始对象和短语；改写失败时使用原问题，保留 warning |
| 元数据词法零命中 | 不阻断正文向量召回 |
| 语义服务失败 | 在原允许范围使用 lexical，返回 semantic_unavailable 与 lexical_fallback |
| 显式约束无论文 | not_found |
| 有允许论文但无最终证据 | insufficient_evidence，禁止生成事实或伪造引用 |
| comparison 目标不足两篇、limit 小于目标数 | invalid_input |
| Catalog 未就绪 | catalog_not_ready |

JEV 超时后先等待再重试，至少执行一次重试；JEV_RETRY_COUNT 表示首次请求之外的重试次数，实际最小值为 1。等待采用 1、2、4 秒的退避。当前配置为 15 秒超时、2 次重试，最多尝试三次，分别等待 1 秒、2 秒；全部失败后才使用本地规则。鉴权等不可重试 HTTP 错误仍直接回退。

## 验收与指标

中文输入、中文答案，英文语料。指标关注目标论文和关键证据覆盖，不要求每次返回满 8 条，也不绑定固定 Chunk 排名：

- 目标论文是否进入允许范围及任务目标。
- 关键证据是否进入最终 top 8。
- reason 窗口能否连续解释机制，是否重复。
- summary 是否覆盖实际存在的方法、实验和结论。
- comparison 双方是否有相关证据，不能用背景摘要冒充具体比较结论。
- Chunk ID、章节、页码和 source_id 是否可追溯。

测试在 RAG_project 环境执行 pytest 与 pip check；真实验收连接 stdio MCP，只提交中文问题，记录实际自动 task、mode、过滤、查询改写、降级、证据和 Codex 回答。不得用模拟答案脚本或 Query Rewriter 模型生成最终答案。数据和向量不在验收时重建。

2026-10-07 的六类真实调用见 record.md 新增章节：元数据返回两篇；正文默认 hybrid，各返回八条，比较任务两篇各四条。表格首条命中 EfficientNet 表 2。LoRA 摘要虽然覆盖关键章节，本次实验来源仍以参数设置为主，未覆盖完整定量结果；记录保留了这一质量限制。章节覆盖不能替代事实覆盖检查。
