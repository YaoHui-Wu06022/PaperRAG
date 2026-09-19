---
name: paper-curate
description: Maintain the Paper RAG source library and research Wiki through explicit ingestion, validation, and revisioned publication.
---

Use `ingest` for PDFs and Markdown notes, `index` for incremental lexical/dense publication, and `doctor` to inspect dependencies and version consistency. Never import generated Wiki pages as primary sources. Before writing a Wiki page, read the cited source evidence; use `wiki-prepare` to inspect dependencies and `wiki-apply` with the expected revision and file hash. Run `wiki-lint` before publishing and preserve conflicts or stale dependencies for review.

Keep paper facts, personal observations, and agent synthesis clearly separated. A lint pass verifies links, IDs, and versions; it does not prove that prose is supported by the cited text. Do not call a project-owned generative LLM service from curation commands.

Read [references/publication.md](references/publication.md) for import requests, page formats, version conflicts, and publication. Source import and Wiki publication require a user request to import or curate; ordinary research should not write its answer back to Wiki. `wiki apply` validates before writing; `wiki lint` checks the published result.
