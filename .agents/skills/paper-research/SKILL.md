---
name: paper-research
description: Use the local Paper RAG library to find, read, compare, and cite papers and notes without delegating answer generation to the library.
---

Use the `paper-rag` JSON CLI as an evidence service. Start with `papers-count` or `papers-find` when the request asks about corpus scope, then use `search` and `read` to retrieve exact evidence. Use `citations` for local citation edges and `wiki-search`/`wiki-read` only as maintained navigation.

Treat every substantive claim as needing an evidence ID. Read the source span before citing it, distinguish paper facts from personal notes and agent synthesis, and state when the local corpus cannot establish a negative claim. Expand searches to appendix, references, or citation neighbors only when the question requires it. The library does not generate answers; synthesize the response in the host agent and preserve the returned version and warnings.

Read [references/commands.md](references/commands.md) for requests and pagination. Keep CLI switches ASCII and put Chinese text in UTF-8 JSON request files. Ordinary research is read-only; publish a Wiki page only when the user asks to save or organize it.
