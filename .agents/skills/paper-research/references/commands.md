# Research requests

Run from the repository root with Conda `RAG_project`:

```text
python -X utf8 -m paper_rag search --request request.json --json
```

`--request -` reads stdin. Every request accepts `schema_version: 1`; unknown
fields are rejected. `--project-root PATH` can precede the command.

| Command | Example JSON request |
| --- | --- |
| `doctor` / `status` | `{}` or `{"remote":true}` for vector verification |
| `papers find` | `{"query":"Attention","limit":20,"offset":0}` |
| `papers get` | `{"document_id":"d_..."}` |
| `papers count` | `{"filters":{"source_kind":"paper","year_min":2020}}` |
| `search` | `{"query":"注意力机制","keywords":["scaled dot product attention"],"mode":"hybrid"}` |
| `read` | `{"evidence_id":"ev:d_...:r_...:b000001:0:500","max_chars":12000}` |
| `read` | `{"document_id":"d_...","list_sections":true}` |
| `read` | `{"document_id":"d_...","section_id":"s_...","include_children":true}` |
| `citations` | `{"document_id":"d_...","direction":"outgoing","depth":1}` |
| `wiki search` | `{"query":"attention"}` |
| `wiki read` | `{"page_id":"attention","offset":0,"max_chars":12000}` |

Search filters: `document_ids`, `source_kind` (`paper` or `note`), `title`,
`author`, `venue`, `year_min`, `year_max`, and `tags`. They are explicit scope
constraints. Default regions are `abstract` and `body`; add `appendix` when
needed. Reference entries come from `citations` or structure reading.

Preserve the original request when following `next_cursor`; metadata, citation,
and Wiki lists instead return `next_offset`. Search pages cover a bounded
candidate pool, not an exhaustive corpus scan. Use more specific query variants
for further coverage. Read pagination covers all requested source text.

Response envelope: `schema_version`, `status`, `data`, `warnings`. Inspect
`degraded`, corpus version, `historical`, and `withdrawn`; never silently use
stale vectors or an old revision as current evidence. Unavailable dense retrieval
falls back to lexical search in hybrid mode. Dense-only mode cannot provide that
fallback. Empty top-k does not establish absence.
