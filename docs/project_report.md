# 面向本地论文库的 RAG 问答系统：项目说明报告

## 1. 项目定位

本项目面向本地论文库的文献知识管理和专业问答场景，目标是把论文原文、章节结构、表格、元数据、引用关系和检索证据组织成一条可追溯的 RAG（Retrieval-Augmented Generation）链路。

系统的核心原则是：

1. 先把论文解析为可检索、可定位、可增量更新的结构化知识单元。
2. 根据问题类型选择元数据查询、正文检索、引用图检索或组合检索。
3. 把召回结果整理为带来源、页码、章节和排序信息的证据上下文。
4. 由上层 Agent 基于证据生成最终回答，系统本身负责检索和证据组织，不直接替代上层 Agent 生成答案。

当前实现主要由 SQLite Catalog、MinerU 解析结果、LlamaIndex/Milvus 向量索引、SQLite FTS5 词法检索、JEV 路由和 MCP 工具层组成。

## 2. 总体技术架构

<img src="https://wuyaohui06022.oss-cn-chengdu.aliyuncs.com/2026/mermaid-diagram.png" style="zoom:67%;" />

查询链路和索引链路相互独立。索引链路负责把文档变成稳定的数据和向量；查询链路负责根据问题选择候选范围、召回证据并返回结构化结果。

## 3. 文档解析与知识库构建

### 3.1 MinerU 解析

论文首先经过 MinerU Pipeline，输出文档块、页码、章节、表格、公式、代码、图片以及原始 HTML 等信息。解析结果保留原始块类型和来源定位，避免在进入检索前丢失版面和文档结构。

系统重点保留以下结构信息：

- 论文标识、标题、作者、年份、分类和文件路径；
- 章节编号、章节标题和完整的 `section_path`；
- 页码范围和原始 block 标识；
- 表格、公式、代码、图片和图表等结构化块；
- References 区域中的参考文献、ArXiv ID、DOI 以及本地混合匹配结果；
- 论文之间的引用边。

### 3.2 Chunk 生成

Chunk 生成逻辑位于 `paper_rag/catalog/chunks.py`。普通正文按照固定规则切分

- 普通文本最大长度为 1200 字符；
- 相邻正文 Chunk 保留 150 字符重叠；
- Abstract、正文 Content、Appendix 分区处理；
- Reference 区域不作为普通正文 Chunk 参与召回；
- 公式、表格、代码、图片、图表、列表等结构化块单独成 Chunk；
- 每个 Chunk 记录前后邻居摘要，用于需要扩展上下文的推理任务。

每个 Chunk 具有稳定的 `chunk_id`、`paper_id`、`canonical_id`、序号、类型、章节、页码、来源 block 和资源引用信息，并计算内容哈希。这样既能在结果中定位原文，也能在文档未变化时复用已有向量。

### 3.3 表格的双表示设计

表格同时保存两种文本：

- `text`：保留原始表格 HTML，便于展示、审计和回溯；
- `retrieval_text`：把表格转换成面向检索的纯文本结构，包含表题、列名和逐行的字段值。

例如，表格的检索文本会被组织为：

```text
Table 2: Performance comparison.

Columns: Model | Accuracy | Params

BERT | Accuracy 89.2 | Params 110M
RoBERTa | Accuracy 91.3 | Params 125M
```

`retrieval_text` 会过滤 `script`、`style` 等无关内容并解码 HTML 实体；解析失败时使用纯文本回退。原始 HTML 仍由 `text` 保留，避免展示层和检索层互相污染。

## 4. SQLite Catalog：结构化数据与增量依据

`paper_rag/catalog/service.py` 负责构建和维护 SQLite Catalog。Catalog 是系统的事实来源，Milvus 只保存向量和必要的索引字段。

核心表包括：

- `papers`：论文元数据和解析状态；
- `chunks`：Chunk 文本、`retrieval_text`、章节、页码、类型和哈希；
- `papers_fts`、`chunks_fts`：SQLite FTS5 词法检索表；
- `references`、`references_fts`：参考文献及其检索索引；
- `citation_edges`：论文之间的引用关系；
- `catalog_meta`：Catalog 版本和构建时间；
- `embedding_state`、`embedding_items`：Embedding 模型、维度、Chunk 哈希、集合名称和同步状态。

Catalog 重建采用临时数据库加原子替换方式。构建失败时保留上一份可用 Catalog，避免半成品数据覆盖当前知识库。

## 5. Embedding 输入和 Milvus 索引

### 5.1 Embedding 输入的确定性

LlamaIndex 节点由 `paper_rag/llamaindex/nodes.py` 从 SQLite Chunk 创建。节点的正文设置为：

```python
node.text = chunk.retrieval_text
```

Embedding 输入严格是 `retrieval_text`。Chunk 的标题、作者、页码、章节、哈希、来源 block 等字段只作为 metadata 保存，并通过 `excluded_embed_metadata_keys` 排除，不会被 LlamaIndex 自动拼接到 Embedding 文本中。因此表格的 Embedding 输入是结构化纯文本，而不是原始 HTML。

节点同时保留原始 `content_text`，用于结果展示和证据引用。检索文本和展示文本分离，使向量质量、审计能力和用户可读性可以分别优化。

### 5.2 DashScope Embedding

`paper_rag/llamaindex/embedding.py` 实现 DashScope `/embeddings` HTTP 客户端和 LlamaIndex 适配器，支持：

- 批量发送文本；
- 指定模型、维度和编码格式；
- 对限流和临时服务错误进行有限重试；
- 校验返回向量数量和向量维度；
- 超时和错误信息向上层传播。

### 5.3 Milvus 生命周期

`paper_rag/llamaindex/index.py` 负责向量索引构建和同步，当前索引模式包括 `auto`、`incremental` 和 `full`。

一次索引更新的主要步骤如下：

1. 读取 Catalog Chunk 和当前索引 Manifest。
2. 使用 `(chunk_id, retrieval_text_hash, model, dimensions, chunk_rule_version)` 判断 Chunk 是否仍可复用。
3. 对未变化 Chunk 复用已有 Milvus 向量。
4. 只对新增或内容变化的 Chunk 调用 Embedding API。
5. 写入 staging collection，并校验记录数和向量维度。
6. 校验成功后原子写入 Manifest 和 Embedding cache。
7. 最后切换到新集合并清理旧集合。

如果新集合构建或校验失败，系统删除 staging collection，恢复旧 cache，并保留上一份可用索引。状态接口会报告 Manifest、Catalog、集合、模型、维度和 cache 是否过期，以及最近一次同步失败原因。

## 6. 问题路由与检索

### 6.1 JEV 路由

上层 Agent 已经决定需要执行 `library_retrieve` 后，系统才调用 JEV 对检索任务分类。当前任务类型包括：

- `FACT`：查找单个事实或直接证据；
- `REASON`：需要同一论文中的相邻或补充上下文；
- `SUMMARY`：按论文组织摘要和正文证据；
- `COMPARISON`：平衡召回多篇论文，支持对比。

JEV 只接收任务分类所需的问题文本，并返回任务类型和置信度。JEV 不负责生成答案，也不决定最终证据内容。JEV 未配置、调用失败或置信度不足时，系统使用本地规则回退，保证检索入口仍可用。

### 6.2 候选论文过滤

系统先在 SQLite Catalog 中解析作者、分类、年份、时间范围和论文状态等元数据条件，得到候选 `paper_id` 集合，再把候选范围传给正文检索。这样可以在进入向量召回前缩小搜索域，避免仅依赖语义相似度处理明确的元数据约束。

### 6.3 词法检索

`SQLiteLexicalRetriever` 使用 `chunks_fts` 执行 SQLite FTS5 检索，返回 Chunk、词法排名和词法得分。对于论文标题、专有名词、方法名、缩写和精确数值，词法检索可以补足纯向量检索的召回缺口。

由于论文库以英文为主，词法检索使用独立的查询预处理分支：原始问题始终直接送入 Milvus 的语义检索；library_search 和 library_retrieve 的词法分支先通过 OpenAI 兼容 Query Rewriter 提取完整 `core_terms`，再对这些短语调用腾讯云机器翻译，将其转换为英文 FTS 查询。Query Rewriter 只返回 `core_terms`，例如“注意力机制”保持为一个完整检索短语，不拆成单词。Query Rewriter 或腾讯云不可用时，词法分支退回原始问题，语义分支不受影响。翻译接口只接收查询短语，不发送论文正文或完整 Chunk；作者、分类、年份和状态等结构化 filters 不参与翻译。

翻译后的查询会经过保守的英文停用词过滤，移除 `a`、`the`、`of`、`what` 等高频功能词，保留 `not`、`without`、`vs` 等可能改变语义的词。同时从原始问题中提取 `BM25`、`RAG`、`Qwen3`、模型名、数字和缩写等技术词并追加到查询，降低翻译对专业实体的破坏。停用词只作用于查询字符串，不修改已有的 `chunks_fts` 索引，因此无需重建论文 Catalog。

检索结果的 `retrieval_debug` 或 `query_debug` 字段记录 `core_terms`、`rewriter_used`、`rewriter_fallback`、`translation_used`、`translation_provider`、`translation_fallback` 和 `stopwords_removed`，用于排查召回问题，但不会记录密钥或完整用户问题。

### 6.4 语义检索

语义检索从 Milvus 召回 Embedding 相近的 Chunk，并支持按候选论文和区域过滤。向量库中的文本对应 Chunk 的 `retrieval_text`，结果 metadata 中同时带回原始展示文本、章节、页码和来源信息。

### 6.5 Hybrid + RRF

`HybridRetriever` 将 FTS5 和 Milvus 结果按 `chunk_id` 合并，使用 Reciprocal Rank Fusion（RRF）计算融合排序：

```text
rrf_score = Σ 1 / (rrf_k + rank)
```

结果会保留 `lexical_rank`、`semantic_rank`、`semantic_score` 和 `rrf_score`，便于调试和解释。语义检索失败或没有结果时，系统回退到词法结果，保证局部服务异常不会让整个检索请求失效。

## 7. 任务化证据组织

检索服务位于 `paper_rag/llamaindex/service.py`，统一入口为 `retrieve`。服务返回证据集合和上下文，不直接生成最终回答。

不同任务的证据组织方式如下：

- **FACT**：优先返回与问题直接匹配的 Chunk；
- **REASON**：在核心 Chunk 周围追加同论文的邻近序号 Chunk，补充定义、实验设置或推理条件；
- **SUMMARY**：优先组织每篇论文的 Abstract，再补充正文证据；
- **COMPARISON**：对多个候选论文分别召回并保持来源平衡，避免结果被单篇论文占满。

最终上下文使用 `[S1]`、`[S2]` 等来源编号，附带论文 canonical ID、页码、章节、检索排序信息和原始 `text`。系统会按字符预算截断上下文，并保留来源标识，方便上层 Agent 输出可追溯引用。

## 8. Citation Graph

解析阶段从 References 中提取 ArXiv ID、DOI 和原始引用文本，按 ArXiv、DOI、标题/作者/年份顺序匹配本地论文。所有原始条目写入 `references`，只有本地高置信目标写入 `citation_edges`；重复目标保留在 `references` 并标记主引用。MCP 引用工具通过 SQLite 图遍历提供：

- 一篇论文引用了哪些论文；
- 哪些论文引用了当前论文；
- 给定论文之间的多跳连接；
- 论文引用关系的局部子图。

当前图检索以有限深度 BFS 为主，最大深度为 3，适合回答“这篇论文引用了什么”“相关工作如何连接”等结构化问题。图结果可以与正文证据一起交给上层 Agent。

## 9. MCP 工具层与任务边界

系统通过 MCP 暴露本地论文库能力。核心工具包括：

- `library_search`：执行词法、语义或混合检索；
- `library_retrieve`：按任务类型组织证据上下文；
- `library_get_metadata`：查询论文元数据；
- `library_job_status`：查询异步任务状态。

可选工具集包括论文获取、MinerU 导入、Catalog 同步、索引管理、全文读取和引用图查询。默认工具集启用引用和全文能力，也可以通过 `PAPER_RAG_TOOLSETS` 控制暴露范围，减少 Agent 的工具 Schema 噪声。

涉及写入的操作要求显式 `confirm`，并通过持久化 JSONL Job Log 记录 `queued`、`running`、`succeeded`、`failed` 和 `interrupted` 状态。进程重启后，未完成的活动任务会被标记为 `interrupted`，便于恢复和排查。

## 10. 失败降级和一致性策略

系统将检索可用性和索引一致性分开处理：

- JEV 不可用时回退本地规则；
- 腾讯云翻译服务超时或失败时，词法查询退回原始问题；
- Milvus 无结果或暂时失败时回退 FTS5；
- 向量更新失败时保留旧集合和旧 cache；
- Catalog 构建失败时保留旧数据库；
- 新索引通过 staging collection 校验后才切换；
- 每个结果携带来源和哈希信息，便于审计当前证据来自哪一版数据。

这些策略保证文档导入、向量更新或外部服务短暂异常时，已有知识库仍可继续提供可解释的检索结果。

## 11. CLI 与验证方式

当前提供以下命令入口：

```text
paper-rag catalog sync
paper-rag catalog status
paper-rag catalog chunks
paper-rag index status
paper-rag index rebuild
paper-rag search
paper-rag retrieve
```

项目测试覆盖 Chunk 生成、表格检索文本、Catalog、LlamaIndex 节点、Embedding 输入、索引状态、检索融合、服务路由、引用图和 MCP 工具。当前测试结果为：

```text
74 passed
```

其中 Embedding 输入相关测试验证了节点正文等于 `retrieval_text`，并验证 metadata 不会被拼接进 Embedding 输入。

## 12. 当前实现的边界

当前系统已经形成“解析—结构化存储—增量向量索引—混合召回—证据组织—上层生成”的完整闭环，但仍有几个可以继续演进的方向：

1. 增加离线检索评测集，分别评估 FTS5、Milvus 和 Hybrid 的召回率、证据命中率与引用正确率。
2. 在 RRF 后加入可配置的轻量重排序和去重，减少相邻 Chunk 或同一论文重复占据结果。
3. 增加证据质量评分，综合来源可靠性、章节类型、页码、检索分数和跨论文一致性。
4. 将 Zotero 等文献管理器的标签、收藏夹和批注同步到 Catalog 元数据层。
5. 对引用图、全文证据和元数据查询建立更细粒度的 Agent 工具策略。
6. 在不改变现有 `retrieval_text` 约定的前提下，为公式、图片和复杂表格增加专门的多模态或结构化检索评测。

这些改进应建立在当前稳定的数据契约之上，尤其要保持 `text` 用于展示、`retrieval_text` 用于检索和 Embedding 的职责分离。

