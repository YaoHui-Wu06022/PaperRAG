# 正文检索实现与指标

本文说明论文库正文检索从问题进入到证据返回的完整链路，重点记录候选论文发现、词法检索、语义检索、Hybrid 融合、确定性排序、区域过滤和调试字段。实际调用输入、响应和异常案例保存在 [record.md](record.md)；项目中对应的主要实现位于 `paper_rag/llamaindex/service.py`、`paper_rag/llamaindex/retrievers.py`、`paper_rag/llamaindex/translation.py` 和 `paper_rag/catalog/service.py`。

## 检索入口与任务路由

`library_search` 和 `library_retrieve` 解决不同问题：

| 入口 | 查询对象 | 典型问题 | 返回内容 |
|---|---|---|---|
| `library_search` | 论文元数据 | 哪些论文与注意力相关、某作者有哪些论文 | 论文标题、摘要、作者、分类、日期和资产路径 |
| `library_retrieve` | 正文 Chunk | LoRA 的核心贡献是什么、GQA 如何减少 KV Cache、比较 LoRA 和 QLoRA | 带 `paper_id`、`chunk_id`、章节、页码和来源的证据 |

正文入口首先规范化 `paper_ids`、`filters`、`regions`、`task`、`mode`、`limit` 和 `max_chars`，然后按以下顺序执行：

```text
用户问题
  -> task 路由（fact / reason / summary / comparison）和一次 Query 准备
  -> 候选论文发现
  -> 区域和 paper_id 过滤
  -> lexical / semantic / hybrid 召回
  -> RRF 与确定性质量排序
  -> summary/comparison 的论文级平衡，或 reason 的证据窗口拼接
  -> source_id、Chunk 正文、页码、章节和评分
```

`task="auto"` 时由路由器选择任务；显式传入 `fact`、`reason`、`summary` 或 `comparison` 时不再重新分类。`library_retrieve` 只负责检索和证据组织，不调用答案生成模型，正文结果由 Agent 组织答案。

正文请求只准备一次 `LexicalQuery`。候选发现、SQLite lexical 检索和窄词法 fallback 复用同一份查询，避免重复调用 Query Rewriter 和翻译服务。

MCP 工具描述保持最小职责边界：`library_search` 负责论文元数据并直接输出 `data.presentation.answer_text`；`library_citation` 负责引用关系并直接输出 `data.presentation.answer_text`；`library_retrieve` 负责正文 Chunk 证据，由 Agent 组织答案。正文各模式的 `presentation.render_policy` 都是 `compose`。

## 区域和 filters 约束

当前正文检索允许 `abstract`、`content` 和 `appendix` 三类区域。`reference` 不进入正文向量检索。普通问题默认使用 `("abstract", "content")`；只有问题中出现 `appendix`、`appendices`、`supplement`、`supplementary` 或“附录”，或者显式传入 `regions=["appendix"]` 时，才把 Appendix 加入候选区域。

区域约束在 lexical 和 semantic 两条路径上同时生效：

- SQLite `search_chunks` 通过 `region IN (...)` 和 `paper_id IN (...)` 过滤。
- Milvus Retriever 通过同样的 `region`、`paper_id` 元数据过滤器，并使用 `AND` 组合。
- 候选论文发现的正文回退也使用过滤后的 `allowed_ids`，因此全文回退不能绕过滤器。
- 已显式传入 `paper_ids` 时，不再通过问题发现新的论文；只在这些论文中检索。

支持的论文过滤字段包括 `author`、`category`、`year`、`year_from`、`year_to` 和 `state`。过滤条件先作用于 Catalog，再传给正文检索，避免出现元数据列表和正文证据来源不一致的情况。

## 候选论文发现

候选发现的目标是先确定“哪些论文值得进入正文检索”，再在这些论文内部找 Chunk。候选发现与 Chunk 排序是两个阶段：前者决定论文集合，后者决定证据顺序。

### 1. Query Rewriter 和词法查询准备

正文入口先调用一次 `prepare_lexical_query`。当 Query Rewriter 开启且配置了 API Key 时，它要求模型只返回一个 JSON 对象：

```json
{"core_terms": ["完整技术短语", "方法名", "模型名"]}
```

`core_terms` 保留完整语义，例如 `grouped-query attention` 应作为一个短语保留，而不是拆成若干普通 OR 词。随后：

1. 对包含中文的核心短语逐条翻译，优先调用腾讯云；腾讯云失败后调用已配置的阿里云备用翻译；
2. 英文方法名、缩写、数字和模型名原样保留；
3. 将核心短语转换为带引号的 FTS5 phrase expression；
4. 记录 Query Rewriter 和翻译是否成功。

翻译服务只接收查询短语，不接收论文正文或 Chunk。Query Rewriter 失败、所有翻译服务失败或返回空结果时，会保留 warning 并回退到原始问题的词法规范化结果。调试字段中的 `translation_provider` 表示实际成功的服务；如果阿里云接管，会同时看到 `translation_provider=aliyun`、`translation_fallback=true` 和 `translation_fallback:aliyun`。语义检索始终接收原始问题，不使用翻译后的字符串。

翻译配置使用项目根目录 `.env`：`TENCENT_TRANSLATE_*` 配置主服务，`ALIYUN_TRANSLATION_ENABLED=true`、`ALIYUN_TRANSLATION_ACCESS_KEY_ID`、`ALIYUN_TRANSLATION_ACCESS_KEY_SECRET`、`ALIYUN_TRANSLATION_REGION_ID` 和 `ALIYUN_TRANSLATION_ENDPOINT` 配置备用服务。密钥只从运行环境读取，不写入日志或调试响应。

### 2. 元数据 FTS 优先

第一路查询调用 `search_catalog`，使用 SQLite `papers_fts`。该 FTS 表包含论文的 `title`、`abstract`、`authors` 和 `categories`，有 FTS 查询时按 SQLite `bm25(papers_fts)` 排序。

候选发现同时执行两类元数据查询：

- 使用完整改写查询得到初始 `metadata_ids`；
- 对提取出的技术实体逐个构造精确 FTS 查询，得到 `entity_ids` 和 `entity_hits`。

实体提取优先识别具有技术形态的字符串，例如：

```text
LoRA          QLoRA          GQA
PagedAttention  FlashAttention  ResNet-50
```

普通词 `attention`、`transformer`、`model`、`network` 等会被排除，避免只因为共享通用词而扩大候选集合。实体匹配在候选选择阶段再次使用词边界规则，允许单复数形式，但不会把 `LoRA` 匹配成更长的普通单词。

### 3. 正文 Chunk 回退和触发条件

正文回退调用 `search_chunks`，查询 `chunks_fts` 的 `retrieval_text`、章节和区域字段。只有以下情况才执行候选 Chunk 回退：

- 元数据没有任何结果；
- 元数据没有覆盖 Query Rewriter 提取的技术实体；
- 问题包含 `Table N`、`Tab. N`、`表 N`、`Figure N`、`Fig. N` 或“图 N”。

Fact 和 Reason 不再因为任务类型自动执行额外的候选 Chunk 搜索；论文候选确定后直接进入正文证据召回。

正文回退不会把普通词命中的论文直接加入目标集合。每个 Chunk 必须满足以下任一条件才会进入 `chunk_exact_hit`：

- 命中技术实体或完整核心短语；
- 命中问题中的 Table/Figure 编号；
- 命中正文候选过滤使用的非通用 focus term。

Chunk 的匹配同时查看 `text` 和 `retrieval_text`。`text` 用于展示和回溯，`retrieval_text` 用于 FTS 和向量化，因此表格的 caption、章节信息和图片相邻正文也能参与候选发现。

### 4. 候选论文选择顺序

候选集合按任务采用以下规则：

1. 存在标题完整实体命中时，优先选择 `title_exact_ids`；
2. 否则优先选择正文中命中实体、完整短语或表图编号的 `chunk_focus_ids`；
3. 再使用摘要/元数据实体命中 `entity_ids`；
4. 最后才使用普通元数据或正文回退结果。

`comparison` 会对候选论文去重，并优先固定标题、实体和元数据命中的目标集合；目标不足两篇时才补充正文 focus 命中的论文。论文级平衡阶段最多保留一篇背景论文，确保比较问题的前几条证据不会被通用背景论文占满。

`summary` 只保留首要高置信候选论文。`fact` 和 `reason` 则把正文实体命中放在候选集合前面，以便后续排序能从同一篇目标论文中获取解释证据。

### 5. 候选发现内部字段

候选发现信息只在服务端内部用于诊断和回退，不放入 `library_retrieve.data`，不会发送给 Agent。字段含义如下：

| 字段 | 含义 |
|---|---|
| `metadata_count` | 初始论文元数据 FTS 返回的去重论文数 |
| `chunk_count` | 正文 FTS 返回的去重论文数 |
| `entity_hits` | 每个技术实体对应的论文 ID 列表 |
| `candidate_match_source` | 每个最终候选的首要来源，如 `title_exact`、`abstract_exact`、`chunk_exact`、`entity`、`metadata` |
| `title_exact_hit` | 标题中完整实体命中的论文 |
| `abstract_exact_hit` | 摘要中实体命中的论文 |
| `chunk_exact_hit` | 正文中实体、短语或表图编号命中的论文 |
| `table_ref` / `figure_ref` | 解析出的表号或图号，没有则为 `null` |
| `selected_paper_ids` | 经过任务规则和去重后的最终候选论文 |
| `fallback_used` | 元数据为空且正文候选回退产生结果时为 `true`；元数据有结果但实体覆盖不足或查询含表/图编号时，另看 `chunk_search_used` |
| `chunk_search_used` | 本次是否执行过正文候选搜索 |
| `lexical_query` | 实际传给 Catalog FTS 的安全查询表达式 |

例如，候选为空时可以区分两种情况：

```json
{
  "candidate_discovery": {
    "metadata_count": 0,
    "chunk_count": 0,
    "entity_hits": {},
    "fallback_used": false,
    "chunk_search_used": true,
    "selected_paper_ids": []
  }
}
```

这里表示元数据没有命中，且全文回退也没有找到实体或完整短语；如果 `chunk_count` 大于零但 `selected_paper_ids` 为空，则应继续检查任务规则或实体过滤，而不是误判为 Catalog 没有论文。

## 词法检索

### SQLite FTS5 数据路径

`SQLiteLexicalRetriever` 通过 `search_chunks` 查询 `chunks_fts`。索引保存每个 Chunk 的 `chunk_id`、`paper_id`、`canonical_id`、`section_path`、`region`、`chapter_title`、`type` 和 `retrieval_text`。真正用于匹配的正文文本是 `retrieval_text`，返回时再根据 `chunk_id` 读取完整 `text`、页码、来源 Block 和资源引用。

这样可以把“可检索文本”和“原始展示文本”分开：表格、公式和图片可以在 `retrieval_text` 中带上 caption 或相邻上下文，但不会破坏原始 Chunk 的类型和内容边界。

### 查询词规范化

没有 Query Rewriter 时，`normalize_lexical_text` 会：

- 把连字符和下划线视作词分隔符；
- 保留英文、数字、模型名和中文连续片段；
- 移除常见英语停用词，并在 debug 中记录 `stopwords_removed`；
- 为中文保留紧凑形式，避免中文按空格切词造成空查询。

`build_fts_query` 把词转换成安全的 FTS5 OR 表达式，并对包含空格的核心短语额外生成带引号的 phrase expression。Query Rewriter 路径使用 `phrases=core_terms`，因此不会把完整短语降级为一组互相无关的普通词。

### 词法 debug

每次 lexical 或 hybrid 请求都会在服务端内部保留以下查询状态；这些字段不放入 Agent 的证据响应：

```json
{
  "lexical_query": "\"grouped query attention\" OR \"GQA\"",
  "translation_used": true,
  "translation_provider": "tencent",
  "translation_fallback": false,
  "rewriter_used": true,
  "rewriter_fallback": false,
  "core_terms": ["grouped-query attention", "GQA"],
  "stopwords_removed": []
}
```

这些字段只描述查询处理状态，不记录论文正文。`translation_fallback=true` 表示主翻译服务失败后使用了备用服务，或所有服务失败后回到了原始查询；`rewriter_fallback=true` 表示 Query Rewriter 不可用或返回异常，但后续仍可能使用本地词法查询完成检索。

## 语义检索

语义检索使用当前 active Milvus collection 和已配置的 Embedding 模型。语义 Retriever 接收原始用户问题，不接收 Query Rewriter 的英文结果，避免翻译改变向量语义。

Milvus 检索的约束与 SQLite 一致：

```text
region IN selected_regions
AND paper_id IN candidate_paper_ids（有候选时）
```

服务层会把语义 `top_k` 扩大到至少 `max(limit * 5, 50)`，先取得足够的候选，再交给 RRF 和任务排序。每个语义结果保留：

- `semantic_rank`：Milvus 返回顺序，从 1 开始；
- `semantic_score`：向量库返回的相似度分数；
- `chunk_id`、`paper_id`、区域、章节和页码。

如果 Milvus 不可用、索引未准备好或 semantic 模式没有结果，服务会记录 `semantic_unavailable:*` 或 `lexical_fallback`，随后使用 SQLite lexical 路径完成请求。索引切换失败时保留上一版 active collection，避免检索入口指向不完整索引。

## Hybrid 融合和 RRF

`HybridRetriever` 按 `chunk_id` 合并 lexical 和 semantic 结果。同一个 Chunk 只建立一个合并项，分别记录它在两条召回列表中的名次：

```text
RRF(chunk) =
    1 / (rrf_k + lexical_rank)     （存在 lexical 命中时）
  + 1 / (rrf_k + semantic_rank)    （存在 semantic 命中时）
```

默认 `rrf_k=60`，实际值来自 `LLAMAINDEX_RRF_K`。只被一条路径召回的 Chunk 仍然保留；同时被两条路径召回的 Chunk 通常拥有更高融合分数。多路分数只用于服务端排序，对 Agent 只暴露最终 `score`，避免把内部检索细节混入答案上下文。

RRF 是主相关性依据。确定性质量特征只在 RRF 基础上做稳定的同分或近邻决策，不引入新的 LLM 重排调用，也不改变底层 Chunk。

## 确定性质量排序

### 排序键

普通查询的排序键依次考虑：

1. `rrf_score`；
2. 完整技术实体或短语命中；
3. 查询指定章节命中；
4. 任务相关区域；
5. 证据类型；
6. 同论文同章节重复惩罚。

对应的 `ranking_features` 示例：

```json
{
  "exact_entity_hit": true,
  "section_exact_hit": false,
  "exact_table_ref_hit": false,
  "exact_figure_ref_hit": false,
  "type_priority": "text",
  "type_score": 3,
  "region_priority": "content",
  "region_score": 3,
  "duplicate_penalty": 0.0,
  "quality_bonus": 0.00245
}
```

`quality_bonus` 由固定的小权重组成：实体命中 `0.0014`、章节命中 `0.0011`、区域优先级乘 `0.00025`、类型优先级乘 `0.00015`。这些值只用于确定性排序，RRF 仍然在排序键的第一位。

### 实体、章节和表图编号

`exact_entity_hit` 使用问题中的技术形态短语在 Chunk 文本和 `section_label` 中进行词边界匹配，支持单复数处理。`section_exact_hit` 除实体命中外，还识别：

- `Section 2.2`；
- `Sec. 2.2`；
- `第 2.2 节`。

当问题含有 `Table N`、`Tab. N` 或“表 N”时，排序键首先比较：

1. `exact_table_caption_hit`；
2. `exact_table_ref_hit`；
3. RRF 分数；
4. 实体和章节命中。

Figure 查询使用相同逻辑。这样 Table 2 存在时，caption 明确写有 Table 2 的 Chunk 会排在 Table 5 前面；即使表格的正文解释被另一个 Chunk 命中，仍可通过 `exact_table_ref_hit` 追踪它与目标表号的关系。

### 区域和任务优先级

区域优先级由任务决定：

| task | 优先区域 | 适用场景 |
|---|---|---|
| `fact` | `content` > `abstract`，Appendix 为 0 | 查找定义、数值、直接事实 |
| `reason` | `content` > `abstract`，Appendix 为 0 | 查找方法解释和因果证据 |
| `summary` | `abstract` > `content`，Appendix 为 0 | 摘要、贡献和方法概览 |
| `comparison` | `content` > `abstract`，Appendix 为 0 | 保证各目标论文有正文或摘要证据 |

Appendix 在排序层的优先级为 0；普通问题在更早的区域选择阶段已经排除它。显式 Appendix 查询时，它会进入候选区域，但仍按用户指定的章节、实体和表图编号排序。

### 证据类型和媒体处理

默认证据类型分数如下：

```text
text / paragraph      3
table / list           2
formula / equation    1
image / figure / chart 0
```

普通问题完成排序后，会把 `image`、`figure`、`chart`、`table`、`formula` 和 `equation` 放到正文 Chunk 之后。因此弱相关图片不会占据 Fact 或 Reason 的前三条解释证据。

当问题明确出现 `figure`、`fig`、`image`、`table`、`tab`、`formula`、`equation`、“图”、“表格”或“公式”时，不执行整体媒体后置。表格和图像类型还会获得额外类型加分；公式仍按 formula/equation 的基础类型分数排序。媒体 Chunk 仍然保留原始 `asset_refs`、caption、页码和章节，便于上层按需展示。

### 重复惩罚

排序过程按 RRF 顺序维护 `(paper_id, section_label)` 计数。同一论文同一章节出现后，后续同章节结果增加 `0.00035` 的 `duplicate_penalty`。该惩罚不删除证据，只降低重复片段连续占据前列的概率，让相邻章节或另一篇目标论文有机会进入结果。

## 任务级证据组织

确定性排序后的 Chunk 还要按任务组织成最终返回项：

- `fact`：直接命中的 content Chunk 优先，abstract 作为补充；
- `reason`：以直接命中 Chunk 为核心，只合并同论文、同章节、相邻 ordinal 的 Chunk；
- `summary`：对首要论文优先保留 abstract，再保留贡献和方法正文；
- `comparison`：先按论文分组，再保证每个目标论文至少保留一条摘要或正文证据，最多补充一篇背景论文。

Reason 窗口在内部保留 `source_chunk_ids`、页码和章节；对 Agent 只暴露必要的 `source_chunk_ids` 和证据正文。窗口合并前后会检查句子或段落边界，避免以 `ion kernel`、`d approach` 等半词开头。图片命中时先返回相邻正文解释，再返回图片本身。

`data.evidence` 中每条证据至少包含：

```json
{
  "text": "LoRA freezes the pretrained model and trains a low-rank update.",
  "paper_id": "2106.09685",
  "chunk_id": "...",
  "type": "text",
  "section_path": ["3", "3.1"],
  "page_start": 4,
  "page_end": 5,
  "source_id": "S1",
  "score": 0.029
}
```

`library_retrieve` 不在服务端拼接答案上下文。每个 `evidence` 项直接携带 Chunk 正文 `text`，并保留 `source_id`、`chunk_id`、页码、章节和评分，Agent 可以据此组织答案和引用。

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

| 阶段 | 失败或无结果 | 行为 | 可观察字段 |
|---|---|---|---|
| Query Rewriter | 超时、非法 JSON、空 core_terms | 使用原始问题 | `rewriter_fallback`、`warnings` |
| 翻译 | 调用失败或空译文 | 使用原始查询或已保留的英文实体 | `translation_fallback`、`translation_provider` |
| 元数据 FTS | 无论文 | 执行正文候选搜索 | `metadata_count`、`chunk_search_used` |
| 正文候选 | 无实体、短语或表图编号命中 | 返回 `not_found`，不把普通词论文当目标 | `chunk_count`、`chunk_exact_hit` |
| Milvus | 异常、未就绪或无语义结果 | 回退 SQLite lexical | `semantic_unavailable:*`、`lexical_fallback` |
| Catalog/索引 | 未就绪 | 不返回部分结果 | `catalog_not_ready` 或 `index_not_ready` |

当 `status="not_found"` 时，Agent 只需根据 `warnings` 和空的 `evidence` 判断没有可用证据；候选发现的细分原因留在服务端诊断中。

## 检索指标

当前回归记录以目标论文和关键证据为单位统计，不只看返回长度：

### 候选论文召回率

```text
Candidate Recall@K =
命中人工标注目标论文的查询数 / 查询总数
```

目标论文必须出现在内部候选集合，否则即使后续正文 Chunk 恰好来自目标论文，也算候选发现失败。

### 关键证据 Recall@K

```text
Evidence Recall@K =
前 K 条结果包含标注关键章节或 Chunk 的查询数 / 查询总数
```

Fact 和 Reason 重点检查正文章节、完整表图 caption 和连续解释窗口；Summary 检查摘要或贡献/方法正文；Comparison 检查每篇目标论文是否至少有一条直接相关证据。

### Comparison Precision@K

```text
Comparison Precision@K =
前 K 条中目标论文直接相关证据数 / K
```

背景论文不计为目标论文证据，最多允许一篇背景论文用于补充。误召回分析使用服务端内部候选来源和排序特征，Agent 响应只保留最终证据和单一 `score`。

### 追溯完整性

每条证据必须能通过 `paper_id -> chunk_id -> Catalog` 找回原始文本，并保留 `section_path`、`page_start/page_end`、`source_id`。Reason 窗口还必须保留完整 `source_chunk_ids`，不能只返回拼接后的文本。

## 当前验证方式

可以使用以下命令查看索引状态并执行正文检索；正文响应的主体只包含 `data.evidence`：

回归问题统一使用中文输入，保留 LoRA、GQA、PagedAttention 等英文技术术语；论文正文语料仍以英文为主，用于验证中文查询的词法翻译、语义召回和混合排序。

```powershell
paper-rag index status --json
paper-rag retrieve "GQA 是什么？" --task fact --mode hybrid --json
paper-rag retrieve "PagedAttention 为什么能改善服务？" --task reason --mode hybrid --json
paper-rag retrieve "概括 LoRA 的核心贡献" --task summary --mode hybrid --json
paper-rag retrieve "比较 LoRA 和 QLoRA" --task comparison --mode hybrid --json
paper-rag retrieve "附录 A 中有什么？" --task fact --regions appendix --json
```

应重点检查：

- 每条 `data.evidence` 是否包含完整 `text`；
- 每条证据是否能通过 `source_id`、`paper_id`、`chunk_id`、章节和页码回溯；
- 每条证据是否包含最终 `score`，而不是内部多路分数；
- 普通问题是否没有 Appendix 和弱相关媒体 Chunk；
- Reason 结果是否保留完整窗口和原始 Chunk 回溯信息。

具体案例、实际响应和已发现的问题见 [record.md](record.md)。
