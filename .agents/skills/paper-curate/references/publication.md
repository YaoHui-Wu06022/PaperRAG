# Source and Wiki publication

From the repository root, run `python -X utf8 -m paper_rag COMMAND --request
request.json --json` in Conda `RAG_project`. All content is UTF-8 JSON;
`--request -` reads stdin. Responses carry status and warnings.

Import a source:

```json
{"sources":[{"path":"data/pdf/paper.pdf","metadata":{"title":"Paper","authors":["Author"]}}],"index":true}
```

An existing MinerU directory can be supplied as `mineru_output`. Notes use `.md`
or `.markdown`. Use `document_id` to attach changed source bytes to an existing
document when its original path no longer identifies it. Identical bytes dedupe.
`{"migrate_legacy":true}` migrates legacy outputs; failures remain in the response.

`index` accepts `{}` for incremental publication, `{"rebuild":true}` for a new
Milvus generation, or `{"rollback_generation":"g_..."}` to restore a ready
generation. `{"lexical_only":true}` publishes without dense coverage and must be
reported as such. Failed imports do not replace the last published source.

`wiki prepare` with `{}` returns pending curation and review jobs. Read the
referenced papers with research commands before writing. Supported page kinds
are `paper`, `concept`, `topic`, and `comparison`.

`wiki apply` request:

```json
{"page_id":"attention","kind":"concept","title":"注意力机制","expected_revision":0,"body":"有证据支持的事实 [原文](ev:d_...:r_...:b000001:0:80)","evidence_ids":["ev:d_...:r_...:b000001:0:80"],"quotes":{},"links":[]}
```

For updates, use the current `expected_revision` from `wiki read`. If the user
edited Markdown, read and preserve those edits and supply `expected_file_hash`
from the read response. Body evidence IDs must match `evidence_ids` exactly;
supplied verbatim `quotes` must occur in those source spans. `[[page-id]]` links
must match `links`, resolve to published pages, and do not replace source evidence.

Write in Chinese by default while retaining English terms and verbatim quotes.
Paper cards cover contribution, method, experiments, and limitations. Concepts
cover definition and applicability; topic reviews preserve disagreements;
comparisons state axes and evidence for each difference. Label agent inference.

`wiki apply` mechanically validates before publication; `wiki lint` with `{}`
checks the published library afterwards. Failed validation prevents publication.
Keep semantic verification separate: the host Agent must read each cited span
and verify that it supports the associated assertion. Stale evidence, changed
linked pages, and interrupted publication need review rather than silent repair.
