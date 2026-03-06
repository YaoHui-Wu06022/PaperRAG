# PaperRAG

PaperRAG is a paper-oriented Retrieval-Augmented Generation system for PDF literature QA. The project is built as an end-to-end engineering system rather than a single “call an LLM API” demo: it includes PDF parsing, structured chunking, incremental indexing, hybrid retrieval, reranking, evidence grounding, evaluation, and a Streamlit UI.

This repository is intended to serve as a resume project, so the README focuses on the technical decisions, retrieval pipeline, and benchmark findings.

## 1. Project Goals

The system is designed for scenarios such as:

- building a searchable paper knowledge base from local PDFs
- asking questions against a private literature collection
- returning grounded answers with evidence snippets and page references
- comparing retrieval strategies on public datasets before applying the same parameters to a private corpus

The core design target is not “chat with PDFs” in the abstract. It is:

- structured paper ingestion
- controllable retrieval quality
- reproducible parameter tuning
- traceable evidence output

## 2. Tech Stack

- Python 3.12
- LangChain
- MinerU Cloud for PDF structural parsing
- HuggingFace embeddings, default `BAAI/bge-m3`
- Milvus / Zilliz Cloud as the main vector backend
- Milvus Lite support for Colab benchmarking
- BM25 lexical retrieval
- Dense + BM25 hybrid retrieval with Reciprocal Rank Fusion (RRF)
- `BAAI/bge-reranker-base` reranking
- Streamlit frontend
- Colab benchmark notebook for Qasper evaluation

## 3. System Architecture

### 3.1 Ingestion Pipeline

The ingestion path is:

```text
PDF
-> MinerU Cloud parsing
-> structured blocks / layout metadata
-> parent documents + chunk documents
-> embeddings
-> Milvus vector index
-> local corpus cache / reference cache
```

Key modules:

- [`ingestion/chunking.py`](E:/Pythonproject/RAG_project/ingestion/chunking.py)
- [`ingestion/embedding.py`](E:/Pythonproject/RAG_project/ingestion/embedding.py)
- [`retrieval/vector_store.py`](E:/Pythonproject/RAG_project/retrieval/vector_store.py)
- [`pipeline.py`](E:/Pythonproject/RAG_project/pipeline.py)

### 3.2 Retrieval Pipeline

The QA retrieval path is:

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

Core modules:

- [`services/retrieval_service.py`](E:/Pythonproject/RAG_project/services/retrieval_service.py)
- [`retrieval/retriever.py`](E:/Pythonproject/RAG_project/retrieval/retriever.py)
- [`retrieval/metadata_filter.py`](E:/Pythonproject/RAG_project/retrieval/metadata_filter.py)
- [`retrieval/query_rewrite.py`](E:/Pythonproject/RAG_project/retrieval/query_rewrite.py)

### 3.3 UI Layer

The frontend uses Streamlit and is focused on evidence visibility rather than only answer text. It supports:

- asking questions against the main corpus
- reference-only retrieval
- evidence snippet display
- source / section / page display
- original page highlighting via bbox coordinates when available

UI entry:

- [`app/streamlit_app.py`](E:/Pythonproject/RAG_project/app/streamlit_app.py)

## 4. Key Technical Design

### 4.1 Semantic Paper Chunking

Chunking is not plain fixed-window splitting. The project uses `semantic_paper` as the main strategy:

- keep semantically coherent parent blocks first
- only fall back to token-based splitting for oversized blocks
- preserve `parent_id`, `structure_block_id`, page, and source metadata

Relevant config:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `CHUNK_STRATEGY=semantic_paper`
- `CHUNK_SEMANTIC_HARD_MAX_CHARS`
- `CHUNK_MIN_BLOCK_CHARS`

Implementation:

- [`ingestion/chunking.py`](E:/Pythonproject/RAG_project/ingestion/chunking.py)

### 4.2 Hybrid Retrieval

The system supports three retrieval modes:

- `dense`
- `bm25`
- `hybrid`

`hybrid` combines dense retrieval and BM25 using RRF. This is useful for paper QA because:

- dense retrieval helps on semantic paraphrase
- BM25 helps on exact terminology, dataset names, model names, metrics, and ablation keywords
- RRF gives a stable fusion without requiring score normalization across retrievers

Implementation:

- [`retrieval/retriever.py`](E:/Pythonproject/RAG_project/retrieval/retriever.py)

### 4.3 Query Rewrite

The project uses a lightweight rule-based query rewrite rather than an LLM rewrite layer. The current implementation generates multiple query variants from:

- the original query
- a stripped version without conversational boilerplate
- a keyword-compressed variant
- cross-lingual term expansion
- academic intent expansion
- acronym expansion such as `RAG`, `LLM`, `QA`, `VQA`, `GNN`

This keeps inference cost low while improving retrieval coverage for academic search queries.

Implementation:

- [`retrieval/query_rewrite.py`](E:/Pythonproject/RAG_project/retrieval/query_rewrite.py)
- [`tests/test_query_rewrite.py`](E:/Pythonproject/RAG_project/tests/test_query_rewrite.py)

### 4.4 Metadata Filtering

The retrieval layer supports explicit metadata constraints in the query, including:

- author
- title
- venue
- year
- source
- keyword

The filter supports exact year, year ranges, relative time forms, and generates both:

- a local post-filter
- a Milvus-side document pre-filter when possible

This makes structured academic search possible without forcing users into a separate advanced search UI.

Implementation:

- [`retrieval/metadata_filter.py`](E:/Pythonproject/RAG_project/retrieval/metadata_filter.py)

### 4.5 Reranking and Evidence Diversity

After retrieval, the system optionally applies a reranker and then performs:

- evidence deduplication
- source diversification
- per-source evidence caps
- comparison-query entity coverage supplementation

This is intended to reduce repeated evidence and improve answer coverage across multiple papers.

Implementation:

- [`retrieval/retriever.py`](E:/Pythonproject/RAG_project/retrieval/retriever.py)
- [`services/retrieval_service.py`](E:/Pythonproject/RAG_project/services/retrieval_service.py)

### 4.6 Incremental Index Maintenance

The project supports incremental updates by `doc_id` instead of rebuilding the whole vector store each time.

Supported flows:

- ingest new paper
- replace an existing `doc_id`
- delete one or more `doc_id`
- keep local corpora and vector store in sync

Key commands:

- `python main.py ingest`
- `python main.py ingest --force`
- `python main.py delete-doc <doc_id_1> <doc_id_2> ...`

Implementation:

- [`retrieval/vector_store.py`](E:/Pythonproject/RAG_project/retrieval/vector_store.py)
- [`services/sync_transaction.py`](E:/Pythonproject/RAG_project/services/sync_transaction.py)

## 5. CLI and Runtime Flow

Main CLI commands:

- `ingest`: build or update the vector index from local PDFs
- `delete-doc`: remove one or more papers by `doc_id`
- `ask`: single-turn QA
- `chat`: multi-turn interactive QA
- `models`: list available AIHubMix models
- `health`: startup validation and dependency checks

Argument definitions are in:

- [`main.py`](E:/Pythonproject/RAG_project/main.py)

Typical local usage:

```bash
python main.py ingest
python main.py ask "What datasets are used for evaluation?"
streamlit run app/streamlit_app.py
```

## 6. Benchmark Workflow

Local CLI is focused on private-corpus ingestion and QA. Public-dataset benchmarking is intentionally moved to Colab:

- benchmark logic: [`colab_eval/dataset_benchmark.py`](E:/Pythonproject/RAG_project/colab_eval/dataset_benchmark.py)
- benchmark notebook: [`notebooks/PaperRAG_Qasper_Eval_Colab.ipynb`](E:/Pythonproject/RAG_project/notebooks/PaperRAG_Qasper_Eval_Colab.ipynb)

The notebook supports:

- cloud Milvus / Zilliz
- Milvus Lite in Colab
- parameter overrides without uploading `.env`
- progress display for corpus preparation and evaluation
- summary JSON + detail CSV export

### 6.1 Benchmark Caveat

The benchmark is a sampled closed-set retrieval benchmark, not a full open-web or full-production benchmark.

For Qasper:

- sampled QA rows are converted into a temporary benchmark corpus
- the system retrieves against that temporary corpus
- metrics are useful for parameter comparison across runs
- metrics should be treated as relative engineering indicators, not final production truth

## 7. Experimental Results

The repository includes benchmark summaries in:

- [`notebooks/evalute/dataset_benchmark_summary_all.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_all.json)
- [`notebooks/evalute/dataset_benchmark_summary_dense.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_dense.json)
- [`notebooks/evalute/dataset_benchmark_summary_bm25.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_bm25.json)
- [`notebooks/evalute/dataset_benchmark_summary_false_reranker.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_false_reranker.json)
- [`notebooks/evalute/dataset_benchmark_summary_false_rewrite.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_false_rewrite.json)

All experiments below use:

- dataset: `qasper`
- samples: `50`
- vector backend: `milvus`
- chunk size: `512`
- overlap: `128`
- final top-k: `5`

### 7.1 Retrieval Mode A/B

| Setting | Hit@5 | MRR@5 | Avg Latency (ms) | P95 Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 0.66 | 0.5667 | 101.93 | 274.67 |
| Dense only | 0.64 | 0.5253 | 94.01 | 221.26 |
| BM25 only | 0.60 | 0.5333 | 12.85 | 28.81 |

Analysis:

- Hybrid gives the best overall retrieval quality on this Qasper slice.
- Dense-only is stronger than BM25 on Hit@5, which suggests semantic similarity helps paper QA.
- BM25 is much faster and still reasonably competitive on MRR, which indicates exact-term matching is still valuable for paper datasets.
- Hybrid is the best quality/robustness choice for the main system, while BM25 can be useful as a low-latency fallback or analysis baseline.

### 7.2 Effect of Reranker

| Setting | Hit@5 | MRR@5 | Avg Latency (ms) |
| --- | ---: | ---: | ---: |
| Hybrid + rewrite + reranker | 0.66 | 0.5667 | 101.93 |
| Hybrid + rewrite + no reranker | 0.66 | 0.5667 | 124.35 |

Analysis:

- On this 50-sample Qasper slice, enabling reranking did not improve Hit@5 or MRR@5.
- The latency difference here is not reliable evidence that reranking is cheaper; the two runs are close enough that runtime variance, Milvus collection variance, and Colab noise are plausible explanations.
- The practical conclusion is that reranking is not yet clearly buying measurable gains on this benchmark configuration.

### 7.3 Effect of Query Rewrite

| Setting | Hit@5 | MRR@5 | Avg Latency (ms) |
| --- | ---: | ---: | ---: |
| Hybrid + rewrite | 0.66 | 0.5667 | 101.93 |
| Hybrid + no rewrite | 0.68 | 0.5380 | 39.47 |

Analysis:

- Query rewrite improved ranking quality as measured by MRR (`0.5380 -> 0.5667`), meaning relevant evidence is ranked earlier on average.
- Disabling rewrite slightly improved Hit@5 on this slice (`0.66 -> 0.68`), so rewrite is not a pure win.
- Rewrite adds noticeable latency because each question is expanded into multiple retrieval variants and fused with RRF.
- The engineering interpretation is that query rewrite is a tradeoff:
  - better early-rank quality
  - higher latency
  - not guaranteed to improve coarse recall on every sample

### 7.4 Current Best Configuration in This Repository

The current “mainline” benchmark configuration stored in [`notebooks/evalute/dataset_benchmark_summary_all.json`](E:/Pythonproject/RAG_project/notebooks/evalute/dataset_benchmark_summary_all.json) is:

- `retrieval_mode=hybrid`
- `retriever_top_k=30`
- `final_top_k=5`
- `query_rewrite_enabled=true`
- `query_rewrite_max_variants=3`
- `use_reranker=true`
- `retrieval_score_threshold=0.05`
- `diversify_by_source=true`
- `max_chunks_per_source=5`

Observed benchmark summary:

- `Hit@5 = 0.66`
- `MRR@5 = 0.5667`
- `Avg retrieval latency = 101.93 ms`
- `P95 retrieval latency = 274.67 ms`

## 8. Why This Is More Than “Calling APIs”

This project contains several engineering concerns that go beyond wiring an LLM endpoint:

- document parsing and structured cache design
- semantic chunking and parent-context reconstruction
- incremental vector index maintenance
- hybrid lexical/semantic retrieval
- query rewriting and metadata-aware retrieval
- reranking and evidence diversity control
- benchmark-driven parameter selection
- evidence-grounded UI and answer traceability

The value of the project is in the retrieval system design, tuning loop, and end-to-end reproducibility.

## 9. Repository Structure

```text
app/            Streamlit UI
colab_eval/     Benchmark implementation for Colab
generation/     LLM client and prompt logic
ingestion/      PDF parsing, chunking, embeddings
retrieval/      vector store, retriever, metadata filter, query rewrite
services/       orchestration, retrieval flow, health checks
tests/          focused tests for retrieval and reference logic
notebooks/      Colab benchmark notebook and result summaries
```

## 10. Environment Configuration

The repository includes a credential-free example file:

- [`.env.example`](E:/Pythonproject/RAG_project/.env.example)

Typical fields:

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

LLM_PROVIDER=aihubmix
LLM_MODEL=gpt-4.1-free
AIHUBMIX_API_KEY=your_key
```

## 11. Resume-Oriented Summary

If you need to summarize this project on a resume, the technically honest framing is:

- Built an end-to-end paper RAG system with MinerU-based PDF parsing, semantic chunking, incremental Milvus indexing, and evidence-grounded QA.
- Implemented dense, BM25, and hybrid retrieval with query rewrite, metadata-aware filtering, reranking, and source diversification.
- Built a Colab-based benchmark workflow on Qasper to compare retrieval configurations and tune parameters with reproducible JSON/CSV outputs.
- Developed a Streamlit interface for grounded literature QA with evidence snippets, page metadata, and traceable source display.
