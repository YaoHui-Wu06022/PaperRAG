# Agent research library implementation

Environment: conda `RAG_project`, Python 3.12. Preserve user changes to AGENTS.md.

## Accepted design

MinerU + Milvus; no project-owned generative model calls. Versioned evidence in
SQLite, FTS5 lexical search, immutable original assets, JSON CLI, research and
curation skills, revisioned Markdown wiki. The two retired legacy collections and
the `paper_rag_chunks` alias were archived and removed; the active v2 collection
is the only Paper RAG dense index.

## Milestones

- [x] S0: source/data/remote index inventory, baseline tests.
- [x] S1: immutable revisions, stable document identity, import and reading.
- [x] S2: lexical/dense retrieval, incremental indexing, CLI contracts.
- [x] S3: research skill, cited research workflow.
- [x] S4: wiki preparation, validation, revision publication and lint.
- [ ] S5 full acceptance: engineering cutover and SQLite scale verification are implemented; remote rollback and host-agent answer-quality acceptance remain open.

## Verification

Use Python from `C:/Users/Fallinty/anaconda3/envs/RAG_project/python.exe` with
`-X utf8`. All shell commands use rtk. UTF-8 requests carry non-ASCII values;
CLI switches and request filenames are ASCII. Run offline tests before live
integration. New Milvus collections use a separate v2 namespace.

Baseline: 37 tests passed. Original corpus: 21 papers, 2375 blocks, 991 chunks.
Old remote collection: 1001 rows, 198 shared IDs have different text, 16 missing
local IDs. Historical evaluation reports are not valid v2 quality baselines.

Remaining external acceptance: create one additional Milvus generation and run a
real remote rollback rehearsal after the Zilliz database collection limit is
raised or old test collections are explicitly retired. This implementation never
deletes those collections automatically.

Observed collections at the quota check: `content_index` (552 rows),
`paper_rag_chunks__staging_20684b514b0c` (838),
`paper_rag_chunks__staging_6f5cf69858a1` (854),
`paper_rag_chunks__staging_fbaab99f121f` (1001), and the active v2 collection
(1050). Names containing `staging` are not proof that deletion is safe; no old
collection or alias was changed.

Quality acceptance is also distinct from tool availability. The same-host-model
single-retrieval vs agent-follow-up answer experiment is not yet graded. The
current hybrid run improves held-out Hit@10 and span Recall@10 against FTS-only,
but MRR (0.469 vs 0.497) and nDCG (0.585 vs 0.597) are lower. This is not a claim
that every retrieval metric improved or that the full no-regression gate passed.

Current live snapshot: 26 active papers, 465 sections, 3,967 blocks, 1,050
chunks, 991 extracted references, 60 conservative local citation edges and 4
Wiki pages. The active v2 generation is verified against Milvus read-after-write
and contains 1,050 rows. `data/library/eval/hybrid.json` records 62 reviewed
fact/structure/comparison/unanswerable cases with no degraded retrieval; the
held-out test split currently reports Hit@10 0.962 and evidence Recall@10 0.952.
Four additional deterministic cases cover metadata counting, a source-checked
citation edge, and complete long-document pagination. All four pass.
Those figures measure retrieval span overlap and do not grade host-agent answer
quality. The benchmark is bound to source and MinerU fingerprints.

The previous synthetic scale fixture was removed during data cleanup; scale
testing remains a separate follow-up and is not represented as a current result.

Current offline suite: 32 passing tests after retiring obsolete routing, answer
client and title-hit evaluation tests. Earlier, all 37 legacy tests and then all
52 combined tests passed before those superseded modules were removed. The new
suite covers publication failure, interrupted Wiki recovery, manual edit
conflicts, stale versions, rollback, cache identity, schema migration, exact
pagination, and versioned-evidence evaluation.

The new `sk-` MinerU credential resolved authentication. The later embedding
`Arrearage` error was resolved by the user's replacement Embedding key; the model
is `qwen3.7-text-embedding-flash`, 1024 dimensions. No credential is included here.
