"""Strict public request contracts. Unicode is carried inside UTF-8 JSON."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1


class Filters(Request):
    document_ids: list[str] | None = None
    source_kind: Literal["paper", "note"] | None = None
    title: str | None = None
    author: str | None = None
    venue: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    tags: list[str] = Field(default_factory=list)


class PapersRequest(Request):
    query: str = ""
    document_id: str | None = None
    filters: Filters = Field(default_factory=Filters)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchRequest(Request):
    query: str = Field(min_length=1, max_length=8000)
    keywords: list[str] = Field(default_factory=list, max_length=8)
    filters: Filters = Field(default_factory=Filters)
    regions: list[Literal["abstract", "body", "appendix"]] = Field(
        default_factory=lambda: ["abstract", "body"]
    )
    mode: Literal["hybrid", "lexical", "dense"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=50)
    candidate_limit: int = Field(default=40, ge=1, le=2000)
    cursor: str | None = None


class ReadRequest(Request):
    document_id: str | None = None
    revision_id: str | None = None
    evidence_id: str | None = None
    section_id: str | None = None
    block_id: str | None = None
    list_sections: bool = False
    include_children: bool = True
    context_before: int = Field(default=0, ge=0, le=20)
    context_after: int = Field(default=0, ge=0, le=20)
    max_chars: int = Field(default=12000, ge=1, le=64000)
    cursor: str | None = None


class CitationRequest(Request):
    document_id: str
    direction: Literal["incoming", "outgoing"] = "outgoing"
    depth: int = Field(default=1, ge=1, le=3)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Source(Request):
    path: str
    document_id: str | None = None
    mineru_output: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestRequest(Request):
    sources: list[Source] = Field(default_factory=list)
    migrate_legacy: bool = False
    index: bool = True
    lexical_only: bool = False


class IndexRequest(Request):
    rebuild: bool = False
    lexical_only: bool = False
    rollback_generation: str | None = None
    rollback_publication: str | None = None
    collection: str | None = None
    archive_digest: str | None = None
    target: str | None = None
    dry_run: bool = False
    force: bool = False


class DoctorRequest(Request):
    remote: bool = False


class WikiRequest(Request):
    query: str = ""
    page_id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    document_id: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=12000, ge=1, le=64000)


class WikiApply(Request):
    page_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    kind: Literal["paper", "concept", "topic", "comparison"]
    title: str = Field(min_length=1, max_length=500)
    expected_revision: int = Field(ge=0)
    expected_file_hash: str | None = None
    body: str = Field(min_length=1, max_length=200000)
    evidence_ids: list[str] = Field(min_length=1, max_length=500)
    quotes: dict[str, str] = Field(default_factory=dict)
    links: list[str] = Field(default_factory=list, max_length=100)


class EvalRequest(Request):
    cases_path: str
    output_path: str | None = None
    mode: Literal["lexical", "hybrid"] = "lexical"
    top_k: int = Field(default=20, ge=1, le=100)
    traces_path: str | None = None
