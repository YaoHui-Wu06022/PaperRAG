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

### 参数与一次请求的边界

| 参数 | 当前行为 |
|---|---|
| query | 非空原始问题；语义查询始终使用它 |
| task | auto/fact/reason/summary/comparison；显式任务跳过 JEV |
| mode | lexical/semantic/hybrid，默认 hybrid；仅为调试切换，不改变工具边界 |
| paper_ids | 可选 ID 列表；按本地 paper/base/canonical ID 解析，任何未满足项均返回 not_found |
| filters | author/category/year/year_from/year_to/state；由 Agent 传入，不由模型补写 |
| regions | 默认 abstract/content；附录提问自动加入 appendix；显式列表仅接受三种正文区域 |
| limit | 最终最多 8 条，输入归一到 1–50；不是上游论文数或原始 Chunk 池大小 |
| max_chars | 默认 24000，当前未用于截断，不能依靠它控制最终文本长度 |

约束解析先用 SQL 确认 Catalog 和可用论文，再进行任务分类、改写和召回。显式 paper_ids 只接受可在本地记录中解析的身份，不根据任意未知版本字符串自动制造记录。没有论文/过滤约束时，两路可查全库允许区域；有 filters 时必须共同遵守过滤范围。

四类任务的不同部分是证据组织，不是不同答案生成模型。MCP 不生成最终正文答案，CLI 也只输出结构化证据。

### JEV 与规则回退

JEV 只接收已经进入 RETRIEVE 的请求，返回四种正文任务之一。routing 保存 route_intent、task、provider、fallback_used、confidence；confidence 来自成功响应，缺失时为 null，不凭空补成 0.9。显式任务标记为 explicit，不调用 JEV。

失败后本地规则依次识别比较、总结、解释/概念/原理，其他默认 fact。单篇“表/图 N 展示/报告/列出”问题优先按 fact 处理，不因“比较结果”误扩大为多篇 comparison。规则回退 confidence 是规则给定的启发式数值，不是 JEV 测得概率。词面规则仍有局限，例如未包含特定解释词的“如何减少”问题不保证总被识别成 reason，需查看实际 routing。

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

当前请求采用 OpenAI 兼容 POST /chat/completions、temperature=0、response_format=json_object，user 内容为 query（最多 7000 字符）和只读 filters。purpose/task 用于内部校验和调试，没有作为额外任务段发送给抽取模型。响应严格校验两个键、数组、非空元素和去重；模型是否完全遵守“从原句抽取”仍靠提示词，不是完整的逐词来源验证。

Rewriter 配置独立于 JEV 和 Agent 的回答模型；禁用或未配置可用凭证时走原问题准备，不代表成功使用了抽取器。JSON 请求成功但返回旧单字段、phrases 或非法数组时，解析失败并回退，不把非法关键词传入 BM25。

### 翻译和阶段查询

原始提取与译文分别保留。仅中文项调用翻译，已有英文名称与缩写保持原样；腾讯云优先，失败时尝试已配置的阿里云。

元数据发现、全库正文召回使用 entities 与 core_terms 的完整短语并集，构造安全的 FTS5 OR 表达式。比较目标确定后，逐篇 BM25 优先使用比较维度，例如 "memory usage"；维度没有词法命中时，仅在同篇论文内回退对象名称。这个回退不再次调用模型或翻译。

Rewriter 超时或非法响应：原问题 → 翻译 → 停用词清理 → BM25。回退普通词不能被当作已经确认的实体。语义检索始终使用原始问题。

翻译尝试 Tencent，再尝试启用且配置凭证的 Aliyun；主服务未配置时也可使用备用。中文项逐项翻译，每个成功查询仅准备一次结果供发现和逐篇召回复用。失败后切换提供商可能重新翻译整组，不能理解为整个请求永远只有一次 HTTP 翻译调用。英文项不请求翻译；随后归一化大小写并清理停用词，完整短语作为一个 FTS 单元，而不是将 mechanism 等普通组成词逐个扩展。

全部翻译失败时保留原词并发 warning，中文词不一定能命中英文库；不能把链路未抛异常理解为译文有效。失败回退可能出现较多普通词，这是未获得合法双字段时的降级结果，应先检查 rewriter_fallback 和 rewriter_error。

纯 semantic 模式当前仍先执行查询准备，因而可能包含 Rewriter/翻译开销和 warning；它不使用改写结果替代向量输入。

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

SQLite chunks_fts 索引 retrieval_text、section_path、region、chapter_title，使用 FTS5/BM25 排序，包含完整章节前缀和表格纯文本；原始表格 HTML text 不单独进入 FTS。papers_fts 索引 base/canonical ID、title、abstract、authors、categories，纯元数据问题不查询 chunks_fts。

Milvus 使用原始问题向量及相同的区域、论文 MetadataFilters。适配器仍做本地范围核对。当前 HybridRetriever 先执行词法再执行语义，不是并行请求。向量/过滤错误进入受控词法回退，不扩大允许论文范围。

同一次请求共用 QueryBundle；首次语义检索生成的问题向量由后续发现和逐篇检索复用。Embedding 失败后不在每篇再次尝试。

按 chunk_id 去重并计算：

```text
RRF = 1 / (rrf_k + lexical_rank) + 1 / (rrf_k + semantic_rank)
```

缺少某路排名时不计该项，默认 rrf_k=60。普通问题以 RRF 为主要排序，实体、章节、区域及类型特征仅用于同分细分。score 为 RRF 分数，不代表事实正确概率。

表格、公式与文本统一参加排序，不整体后置、不给额外类型降权。明确表号、图号匹配优先于一般相关性；问题明确指向某篇论文时，唯一标题匹配仅用于编号查询的排序优先，不改变全库召回范围。普通问题仍保留图片、图表的后置处理。表格原始 HTML、检索纯文本和公式 LaTeX 不变。

### 三种分数与两种排名

| 字段 | 含义与位置 |
|---|---|
| lexical_rank | SQLite BM25 路的名次，从 1 开始；debug 中保留，不是原始 BM25 值 |
| semantic_rank | 语义路的名次，从 1 开始；debug 中保留 |
| semantic_score | 向量库返回的原始相似度分数，不直接与 BM25 相加 |
| rrf_score | 根据两路名次计算的融合值；缺一路只加另一路项 |
| retrieval_rank | 类型/编号排序处理后，该阶段候选池中的顺序 |
| evidence.score | 普通来源的 RRF 值；reason 合并窗口及无排名的章节补充可能为 null |

SQLite 原始 bm25 值按升序选择，但 API 记录的 lexical_rank 是名次。默认 rrf_k=60，两路均第一约为 0.03279，只有一路第一约为 0.01639。lexical/semantic 单路模式也用单路 RRF，而非直接输出 BM25 或原始向量分数。score 不表示事实可信度。

明确表号/图号和图片后置会影响实际顺序，不能仅凭 score 值重新排序响应；某些质量特征仅用于排序细分或调试，并未额外加到 RRF 分数。rankings 描述任务组织前的池，summary 补充和 reason 合并之后不存在与之逐条一一对应的最终排名。

### 初次召回为空时

仅当初次融合池为空，服务才用已准备核心短语、技术名称、表号/图号等做受控词法重查；不调用新模型，不重新拆大量普通词，不扩大论文和区域范围。debug 记录 evidence_fallback_used、evidence_fallback_query、evidence_fallback_reason=primary_retrieval_empty。词法为空但向量已经有结果，不属于整体零召回；语义模式没有结果时也可能回退词法，须读取 warning 和实际数量。

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

摘要优先表示覆盖 abstract 区域，方法/实验/结论用现有章节标题关键词分类，不代表每篇都有对应类别。某个章节首块是配置说明而不是定量实验时，自动补充仍可能缺少问题所需数字；Agent 必须按实际内容作答，不因类别叫 results 就宣称取得完整实验结论。明确的 summary 目标数超过 limit 也返回 invalid_input。

### comparison

每篇分别召回、分别计算论文内 RRF，复用一次查询改写、翻译和问题向量。优先使用比较维度匹配的正文；配额至少两条时保留一条摘要背景。维度词法为空时，仅在同篇用对象名回退。

平均分配名额，余数按目标顺序分配，某篇不足的名额轮流补给其他论文。缺少有效 Chunk 的论文返回 no_evidence_for_paper warning，不能用另一篇替代结论。按论文轮流输出证据，最后统一分配 source_id。

## 返回证据与调试

data.evidence 保留现有字段：source_id、paper_id、chunk_id、text、type、section_path、page_start、page_end、score；reason 另含 source_chunk_ids。页码沿用解析器的零基编号。

返回结构示意如下，内容和数值为接口说明，不是新测试记录：

```json
{
  "status": "ok",
  "data": {
    "query": "用户原问题",
    "task": "reason",
    "mode": "hybrid",
    "routing": {},
    "retrieval_debug": {},
    "evidence": [],
    "count": 0,
    "presentation": {
      "template_version": "library-answer-v1",
      "answer_type": "rag_evidence",
      "render_policy": "compose",
      "answer_text": "",
      "agent_instruction": {
        "version": "rag-agent-v1",
        "task": "reason",
        "system_prompt": "基础规则与当前任务规则"
      }
    }
  },
  "warnings": [],
  "read_only": true
}
```

这里空 evidence 仅表示省略示例内容，真实零证据应返回 insufficient_evidence。接口没有 items、context_text、answer_contract 或 citation_registry，不能再按旧结构取证。完整 Chunk 中的 canonical_id、region、ordinal、资源引用和 retrieval_text 没有全部复制进紧凑 evidence；按 chunk_id 调用 library_get_chunk 查看。窗口来源用 source_chunk_ids 追溯，source_id 只在本次响应内有效。

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

JSON 中 lexical_query 的 `\"` 是字符串引号转义，解码后的 FTS 表达式不含这些反斜杠。例如 JSON 字符串 `"\"attention\""` 的实际值是 `"attention"`，代表一个完整检索短语。它不是翻译或模型额外输出的噪声词。

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

每次指令只拼接基础规则和当前 task 的一段规则，不把四类任务提示词一起发给 Agent。agent_instruction 仅有 version/task/system_prompt，不重复 language、citation_syntax、evidence_field，也不复制证据正文。客户端如需组装模型消息，可将该指令置于自己的系统上下文，将 query 和按 S1/S2 排列的 evidence 放入用户资料区；服务不会自动将工具返回转换成宿主系统消息。

事实依据是证据 text；section_path 和页码用于定位，score 与排名用于调试。不能根据某个 Chunk 命中“实验”章节就补写证据未出现的结果数字。摘要和比较应整合不同来源而非逐条复述，分别引用真正支持该陈述的 [S#]。服务没有答案校验或自动补充模型，Agent 的最终回答由客户端实际大模型完成。

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

当前顶层状态不是单纯的“成功/失败”布尔值。语义失败且有词法证据时，服务可能返回 index_not_ready、milvus_unavailable 或 embedding_unavailable，并保留 data.evidence；客户端要同时看 count/warnings。语义错误 warning 常为 semantic_unavailable:<异常>、lexical_fallback，服务再根据异常文本归类顶层状态。

最终证据为空会覆盖上述降级状态为 insufficient_evidence，并加入 no_evidence_chunks；不应假装有 [S1]。not_found 只用于未满足论文范围，不能把检索没有命中当成论文不存在。Rewriter warning 使用 query_rewriter_failed:<异常>，翻译仍用 translation_failed:tencent/aliyun，并以 translation_fallback:aliyun 标记备用切换。

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

已执行的真实结果见 record.md，具体数量和排名只代表执行时快照。章节覆盖不能替代事实覆盖检查，摘要中有“实验”来源也未必覆盖完整定量结论；须阅读实际 Chunk，而不是根据来源标签判断答案完整性。

## CLI 查看与实现位置

中文问题保存到 UTF-8 JSON，通过 Python argv 列表调用 CLI 的完整方式见 [mcp.md](mcp.md)。CLI search/retrieve 当前始终打印 JSON；CLI 不执行 Agent 回答，读取 evidence 和 agent_instruction 后由真实宿主组织答案。

检索调试应依次看 routing（是否 JEV/规则）、query_debug（抽取/翻译）、constraint_paper_ids/target_paper_ids（范围）、recall（两路与融合数量及排名）、reason_windows/chapter_supplements（组织变化）、最终 evidence/count。四条 reason 证据可能由多块合并产生，不代表 Milvus 只返回四条；limit=8 是上限，不要求凑满。

| 位置 | 内容 |
|---|---|
| paper_rag/mcp/tools/query.py | 简短工具描述和 MCP 入口 |
| paper_rag/llamaindex/service.py | 输入、范围、目标、任务组织、紧凑证据和状态 |
| paper_rag/llamaindex/retrievers.py | SQLite/Milvus 召回、RRF、编号和类型排序 |
| paper_rag/llamaindex/query_rewriter.py | 统一抽取提示词、双字段契约与 HTTP 调用 |
| paper_rag/llamaindex/translation.py | 改写后的翻译、主备提供商、停用词和调试 |
| paper_rag/routing/jev.py、router.py、rules.py | 正文任务、超时等待重试和本地回退 |
| paper_rag/presentation.py | 仅当前任务的 Agent 指令及无证据约束 |

本轮只更新代码行为说明，未执行 Catalog 同步、Embedding 更新或 Milvus Collection 切换。
