# RAG PDF Knowledge Base (MinerU Cloud + Zilliz Cloud)

面向简历项目的论文知识库 RAG 系统，重点是工程结构、检索质量和可追溯性，而不是单纯“能问答”。

## 1. 技术栈

- Python 3.12
- LangChain
- MinerU Cloud（PDF 解析）(从PyMuPDF)
- HuggingFace Embedding（默认 `BAAI/bge-m3`）
- Zilliz Cloud / Milvus（向量库，支持增量 upsert / delete-doc）
- BM25 + Dense Hybrid Retrieval（RRF 融合）
- Reranker（`BAAI/bge-reranker-base`）
- 对比问句实体覆盖检索（如 `A和B`，优先保证双方都有证据）
- Streamlit UI（证据展开 + 原文页 bbox 高亮）

## 2. 系统流程

```text
PDF
 -> MinerU Cloud 解析
 -> block/chunk 结构化 JSONL
 -> Embedding
 -> Zilliz Cloud + BM25 检索
 -> Hybrid 融合 + 条件 Reranker
 -> LLM 严格 JSON 输出
 -> 程序渲染为稳定文本
 -> Sources + 可追溯字段展示
```

## 3. 关键能力

### 3.1 可读引用 + 机器追溯

- 可读引用格式：`[作者简写或标题, 年份, p.<page>, <section_path>]`
- 机器追溯字段：`doc_id/block_id/page/bbox/source/section_path`

### 3.2 Evidence 去重与多样性约束

- 精确去重：按 `doc_id + block_id`
- 结构多样性：同文档 `doc_id + page + section_path` 默认最多 1 条
- 若约束后证据不足，会自动回填（仅去重）

### 3.3 结构化输出协议（JSON）

LLM 被要求只输出：

```json
{
  "conclusion": "1-3句总结",
  "evidence_points": ["证据点1", "证据点2"],
  "uncertainties": ["不确定项1"]
}
```

系统会将 JSON 渲染为最终文本。  
若模型未严格按 JSON 输出，会自动回退到自然语言清洗路径。

### 3.4 检索打分校准

支持三种阈值模式（用于检索和 rerank）：

- `absolute`：固定阈值
- `relative`：相对 top 分数动态阈值
- `quantile`：按分位数动态阈值

这能避免不同查询/不同模型下分数尺度不稳定导致的误拒答。

### 3.5 Streamlit 证据点击高亮

在每条证据中可点击“显示原文页码并高亮 bbox”，展示：

- PDF 对应页图像
- 证据框（bbox）红框高亮
- 原文片段与追溯字段

## 4. 增量入库（Zilliz Cloud / Milvus）

按 `doc_id` 进行增量策略（不再依赖 `MILVUS_DROP_OLD=true`）：

- 新 `doc_id`：插入
- 已存在 `doc_id` 且 `--force`：先删后写
- 同步更新：
  - `data/cache/chunk_corpus.jsonl`
  - `data/cache/parent_corpus.jsonl`
  - `data/cache/block_structured.jsonl`
  - `data/cache/chunk_structured.jsonl`
  - `data/cache/reference_chunk_corpus.jsonl`
  - `data/cache/references_keyword_index.jsonl`

## 5. 删除文档

```bash
python main.py delete-doc <doc_id_1> <doc_id_2> ...
```

会同时删除向量条目和本地索引语料，并清缓存。

## 6. 快速开始

### 6.1 安装

```bash
pip install -r requirements.txt
```

### 6.2 环境变量

仓库内提供了不含密钥的 `.env.example` 示例文件，用于上传到 GitHub。
本地运行时只需要保留你自己的 `.env`，不要提交真实密钥。

```env
# MinerU Cloud
MINERU_API_TOKEN=your_token
MINERU_API_BASE_URL=https://mineru.net/api/v4
MINERU_CLOUD_MODEL_VERSION=pipeline
MINERU_CLOUD_POLL_INTERVAL_SEC=5
MINERU_CLOUD_TIMEOUT_SEC=900

# Embedding
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-m3

# Vector Store
VECTOR_BACKEND=milvus
MILVUS_URI=https://your-zilliz-endpoint
MILVUS_TOKEN=your_token
MILVUS_COLLECTION=rag_pdf_chunks
MILVUS_REFERENCES_COLLECTION=rag_pdf_references
REFERENCES_STRATEGY=keyword_only

# Retrieval + Calibration
RETRIEVAL_MODE=hybrid
RETRIEVAL_SCORE_THRESHOLD=0.2
RETRIEVAL_SCORE_THRESHOLD_MODE=relative
RETRIEVAL_SCORE_RELATIVE_RATIO=0.7
RETRIEVAL_SCORE_QUANTILE=0.6

# Reranker + Calibration
USE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_SCORE_THRESHOLD=none
RERANK_CONDITIONAL_ENABLED=true

# LLM (AIHubMix)
LLM_PROVIDER=aihubmix
LLM_MODEL=gpt-4.1-mini-free
AIHUBMIX_API_MODE=chat
AIHUBMIX_API_KEY=your_key
AIHUBMIX_BASE_URL=https://aihubmix.com/v1
```

当前主流程面向 Zilliz Cloud / Milvus 云端部署，不再依赖 `MILVUS_DROP_OLD` 做全量覆盖。
增量更新按 `doc_id` 走 upsert / delete，同步维护本地 cache 文件。

### 6.3 入库与问答

```bash
# 入库（全量）
python main.py ingest

# 强制覆盖已存在 doc_id
python main.py ingest --force

# 提问
python main.py ask "ResNet和Transformer的创新共通点有哪些?"
```

### 6.4 UI

```bash
streamlit run app/streamlit_app.py
```

### 6.5 Colab 跑 Benchmark

本地主程序不再提供任何 `eval` / `eval-dataset` 命令。

如果你想测试当前参数在 Qasper 数据集上的效果，直接导入 [notebooks/PaperRAG_Qasper_Eval_Colab.ipynb](notebooks/PaperRAG_Qasper_Eval_Colab.ipynb) 到 Colab，从上到下运行即可。这个 notebook 会：

- 克隆仓库
- 安装 benchmark 依赖
- 在 notebook 里直接覆盖环境参数，不要求上传 `.env`
- 输出 `evalout/`，包含和当前项目一致格式的 summary/detail 文件

当前推荐分工：

- 本地：调参数后，对你自己的论文知识库做 `ingest / ask / chat`
- Colab GPU：用 notebook 快速测试这些参数在 Qasper 上的效果，再回本地修改 `.env`
