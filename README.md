# PaperRAG

PaperRAG 是一个面向论文 PDF 知识库的 RAG 系统。项目重点不是“接一个大模型接口做问答”，而是围绕论文检索场景，把文档解析、切块、增量入库、混合检索、重排、证据追踪和评测流程做成一套可调参与可复现的工程系统。

这个仓库按“可写进简历的工程项目”来组织，因此 README 会重点说明：

- 系统目标与应用场景
- 关键技术设计
- 检索链路细节
- Benchmark 方法与实验结论
- 如何将公开数据集上的参数实验迁移到私有论文知识库

## 1. 项目定位

这个项目要解决的问题是：给定一批本地论文 PDF，构建一个可检索、可追溯、可评测的论文知识库问答系统。

目标能力包括：

- 从本地 PDF 批量构建论文知识库
- 对私有论文集合进行问答和多轮对话
- 返回带证据片段、页码、章节信息的答案
- 用公开数据集做检索参数实验，再把结论迁移到私有知识库

相比通用的“Chat with PDF”项目，这个系统更强调：

- 结构化论文解析
- 检索质量可控
- 参数调优可复现
- 证据结果可追踪

## 2. 技术栈

- Python 3.12
- LangChain
- MinerU Cloud：PDF 结构化解析
- HuggingFace Embedding：默认 `BAAI/bge-m3`
- Milvus / Zilliz Cloud：主向量库
- Milvus Lite：Colab 上的轻量 Benchmark 向量库
- BM25：词法检索
- Dense + BM25 Hybrid Retrieval + RRF 融合
- `BAAI/bge-reranker-base`：重排序
- Streamlit：前端展示
- Colab Notebook：Qasper 评测与参数实验

## 3. 系统结构

### 3.1 入库链路

系统的入库流程如下：

```text
PDF
-> MinerU Cloud 解析
-> 结构化 block / layout metadata
-> parent document + chunk document
-> embedding
-> Milvus 向量索引
-> 本地 chunk / parent / reference cache
```

核心代码：

- `ingestion/chunking.py`
- `ingestion/embedding.py`
- `retrieval/vector_store.py`
- `pipeline.py`

### 3.2 问答检索链路

问答时的检索流程如下：

```text
question
-> metadata filter
-> query rewrite
-> dense / bm25 / hybrid retrieval
-> score filtering
-> optional reranker
-> dedupe + diversify by source
-> final evidence docs
-> optional parent-context expansion
-> LLM generation
```

核心代码：

- `services/retrieval_service.py`
- `retrieval/retriever.py`
- `retrieval/metadata_filter.py`
- `retrieval/query_rewrite.py`

### 3.3 前端展示

前端使用 Streamlit，重点不是做一个聊天气泡，而是把证据显示出来。当前支持：

- 主知识库问答
- 参考文献专用检索
- 证据片段展示
- 来源、章节、页码展示
- 在有 bbox 时高亮原始 PDF 页面区域

入口文件：

- `app/streamlit_app.py`

## 4. 关键技术设计

### 4.1 语义块优先的论文切块

项目默认不是简单的固定窗口切块，而是使用 `semantic_paper` 策略：

- 优先保留语义完整的 parent block
- 只有当 block 过长时才回退到 token 级切分
- 保留 `parent_id`、`structure_block_id`、页码、来源等元数据

这比单纯 `chunk_size=xxx` 的方案更适合论文类文档，因为论文往往天然存在章节、段落、公式区和参考文献区的结构边界。

相关配置：

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `CHUNK_STRATEGY=semantic_paper`
- `CHUNK_SEMANTIC_HARD_MAX_CHARS`
- `CHUNK_MIN_BLOCK_CHARS`

实现位置：

- `ingestion/chunking.py`

### 4.2 Dense / BM25 / Hybrid 三种检索模式

系统支持三种检索模式：

- `dense`
- `bm25`
- `hybrid`

其中 `hybrid` 通过 RRF 融合 dense 与 BM25 结果。这个设计对论文问答尤其重要：

- Dense 检索更适合语义改写和抽象表述
- BM25 更擅长命中精确术语、模型名、数据集名、指标名
- RRF 不需要对不同检索器的分数做强归一化，工程上更稳

实现位置：

- `retrieval/retriever.py`

### 4.3 规则式 Query Rewrite

项目没有使用 LLM 做 query rewrite，而是实现了一套轻量规则式改写，用于控制成本并避免额外幻觉。

当前改写会生成多个 query variant，来源包括：

- 原始问题
- 去掉套话后的问题
- 关键词压缩版本
- 中英术语扩展
- 学术意图扩展
- 常见缩写扩展，例如 `RAG`、`LLM`、`QA`、`VQA`、`GNN`

这套设计的目标不是“智能改写得像人”，而是让检索器拿到更适合召回的查询表达。

实现位置：

- `retrieval/query_rewrite.py`
- `tests/test_query_rewrite.py`

### 4.4 Metadata Filter

项目支持在 query 中显式带元数据约束，例如：

- author
- title
- venue
- year
- source
- keyword

过滤器支持：

- 精确年份
- 年份范围
- 相对时间表达
- 本地后过滤
- Milvus 端可下推的预过滤

这样用户可以用接近学术检索的方式提问，而不是只能输入纯自然语言描述。

实现位置：

- `retrieval/metadata_filter.py`

### 4.5 Reranker + 证据多样性控制

召回后系统支持 reranker，并在此基础上做：

- 证据去重
- 来源多样性控制
- 每篇文档最多保留若干条 chunk
- 比较类问题的实体覆盖补召回

这部分是为了避免回答被单篇文档占满，或者返回多条高度重复的证据。

实现位置：

- `retrieval/retriever.py`
- `services/retrieval_service.py`

### 4.6 按 `doc_id` 的增量入库

项目支持按 `doc_id` 做增量更新，而不是每次重建全量向量索引。

支持的操作包括：

- 新论文入库
- 已存在 `doc_id` 的覆盖更新
- 按 `doc_id` 删除论文
- 本地 cache 与向量库同步维护

关键命令：

- `python main.py ingest`
- `python main.py ingest --force`
- `python main.py delete-doc <doc_id_1> <doc_id_2> ...`

实现位置：

- `retrieval/vector_store.py`
- `services/sync_transaction.py`

## 5. 本地运行方式

主程序当前保留的命令包括：

- `ingest`：构建或更新论文向量索引
- `delete-doc`：按 `doc_id` 删除论文
- `ask`：单轮问答
- `chat`：多轮对话
- `models`：列出 AIHubMix 可用模型
- `health`：检查启动依赖与配置状态

参数定义在：

- `main.py`

典型本地使用方式：

```bash
python main.py ingest
python main.py ask "这篇论文用了哪些数据集做实验？"
streamlit run app/streamlit_app.py
```

## 6. Benchmark 工作流

本地主程序主要用于私有论文知识库的入库和问答。公开数据集 Benchmark 被拆到了 Colab：

- Benchmark 主逻辑：`colab_eval/dataset_benchmark.py`
- Colab Notebook：`notebooks/PaperRAG_Qasper_Eval_Colab.ipynb`

Notebook 支持：

- 云端 Milvus / Zilliz
- Colab 上的 Milvus Lite
- 不上传 `.env`，直接在 notebook 内覆盖参数
- 评测进度显示
- 输出 summary JSON 和 detail CSV

### 6.1 Benchmark 的解释边界

这里的 Benchmark 是“抽样闭集检索评测”，不是完整生产环境评测。

以 Qasper 为例：

- 先从数据集里采样若干 QA
- 将样本上下文构成临时评测语料库
- 再在这个临时语料库上做检索

因此这些指标非常适合做参数对比，但不应直接当成线上最终效果。

## 7. 实验结果与分析

仓库中保存了多组 Benchmark 结果：

- `notebooks/evalute/dataset_benchmark_summary_all.json`
- `notebooks/evalute/dataset_benchmark_summary_dense.json`
- `notebooks/evalute/dataset_benchmark_summary_bm25.json`
- `notebooks/evalute/dataset_benchmark_summary_false_reranker.json`
- `notebooks/evalute/dataset_benchmark_summary_false_rewrite.json`

以下实验统一条件：

- 数据集：`qasper`
- 样本数：`50`
- 向量后端：`milvus`
- `chunk_size=512`
- `chunk_overlap=128`
- `final_top_k=5`

### 7.1 检索模式 A/B：Dense / BM25 / Hybrid

| 配置 | Hit@5 | MRR@5 | 平均延迟(ms) | P95(ms) |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 0.66 | 0.5667 | 101.93 | 274.67 |
| Dense Only | 0.64 | 0.5253 | 94.01 | 221.26 |
| BM25 Only | 0.60 | 0.5333 | 12.85 | 28.81 |

结果分析：

- `Hybrid` 取得了三者中最好的综合检索质量。
- `Dense Only` 的 Hit@5 高于 BM25，说明语义召回在论文问答场景中仍然有效。
- `BM25 Only` 的速度明显最快，说明精确术语匹配在论文检索中仍然有价值。
- 当前主系统选择 `Hybrid` 是一个偏质量优先的工程决策，而不是速度优先。

### 7.2 Reranker 的影响

| 配置 | Hit@5 | MRR@5 | 平均延迟(ms) |
| --- | ---: | ---: | ---: |
| Hybrid + Rewrite + Reranker | 0.66 | 0.5667 | 101.93 |
| Hybrid + Rewrite + No Reranker | 0.66 | 0.5667 | 124.35 |

结果分析：

- 在这组 50 样本 Qasper 实验中，打开 reranker 没有进一步提升 Hit@5 或 MRR@5。
- 当前结果不能说明“reranker 一定更快”，因为 Colab 运行时延迟波动和临时 Milvus 集合差异会带来噪声。
- 更稳妥的结论是：在当前配置下，reranker 的收益尚未被这组实验显著验证出来。

### 7.3 Query Rewrite 的影响

| 配置 | Hit@5 | MRR@5 | 平均延迟(ms) |
| --- | ---: | ---: | ---: |
| Hybrid + Rewrite | 0.66 | 0.5667 | 101.93 |
| Hybrid + No Rewrite | 0.68 | 0.5380 | 39.47 |

结果分析：

- 开启 Query Rewrite 后，MRR 从 `0.5380` 提升到 `0.5667`，说明相关证据平均排位更靠前。
- 关闭 Rewrite 时，Hit@5 略高于开启 Rewrite 的结果，因此 Rewrite 不是对所有问题都绝对增益。
- Rewrite 会带来额外延迟，因为系统需要对多个 query variant 分别检索再做融合。
- 当前工程上的理解是：Query Rewrite 提升了排序质量，但存在速度成本，而且不保证粗粒度召回一定上升。

### 7.4 当前主线配置

当前仓库中主线配置对应的结果位于：

- `notebooks/evalute/dataset_benchmark_summary_all.json`

配置为：

- `retrieval_mode=hybrid`
- `retriever_top_k=30`
- `final_top_k=5`
- `query_rewrite_enabled=true`
- `query_rewrite_max_variants=3`
- `use_reranker=true`
- `retrieval_score_threshold=0.05`
- `diversify_by_source=true`
- `max_chunks_per_source=5`

观测结果：

- `Hit@5 = 0.66`
- `MRR@5 = 0.5667`
- `平均检索延迟 = 101.93 ms`
- `P95 检索延迟 = 274.67 ms`

## 8. 仓库结构

```text
app/            Streamlit 前端
colab_eval/     Colab Benchmark 逻辑
generation/     LLM 客户端与 Prompt 逻辑
ingestion/      PDF 解析、切块、Embedding
retrieval/      向量库、检索器、过滤器、Query Rewrite
services/       检索编排、健康检查、同步事务
tests/          检索与参考文献相关测试
notebooks/      Colab Notebook 与实验结果
```

## 9. 环境变量

仓库中提供了去密钥版本的示例配置：

- `.env.example`

典型配置项包括：

```env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-m3

VECTOR_BACKEND=milvus
MILVUS_URI=https://your-zilliz-endpoint
MILVUS_TOKEN=your_token
MILVUS_COLLECTION=rag_pdf_chunks

CHUNK_SIZE=512
CHUNK_OVERLAP=128
CHUNK_STRATEGY=semantic_paper
CHUNK_MIN_BLOCK_CHARS=180

RETRIEVAL_MODE=hybrid
RETRIEVER_TOP_K=30
FINAL_TOP_K=5
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MAX_VARIANTS=3
USE_RERANKER=true
```

## 11. 适合写在简历里的项目描述

如果把这个项目写进简历，比较准确的技术表述可以是：

- 设计并实现面向论文 PDF 的 RAG 系统，支持结构化解析、语义切块、增量 Milvus 入库和证据追踪式问答。
- 实现 Dense、BM25 与 Hybrid 检索链路，并加入 Query Rewrite、Metadata Filter、Reranker 与来源多样性控制。
- 基于 Qasper 构建 Colab Benchmark 工作流，对检索模式和关键参数进行 A/B 实验，并输出可复现的 JSON/CSV 结果。
- 开发 Streamlit 前端，支持论文问答、引用证据展示、页码定位和原文高亮。
