from __future__ import annotations

import pytest

from paper_rag.mcp.toolsets import (
    DEFAULT_ON,
    UnknownToolsetError,
    resolve_enabled,
)
from paper_rag.config import Settings


def test_toolsets_default_and_explicit_profiles():
    assert resolve_enabled("") == set(DEFAULT_ON)
    assert resolve_enabled("none") == set()
    assert resolve_enabled("acquisition,ingestion") == {"acquisition", "ingestion"}
    assert resolve_enabled("all,-management") == {"acquisition", "ingestion", "search-admin"}


def test_unknown_toolset_fails_loudly():
    with pytest.raises(UnknownToolsetError):
        resolve_enabled("not-a-toolset")


def test_settings_loads_toolsets_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text("PAPER_RAG_TOOLSETS=none\n", encoding="utf-8")
    assert Settings.load(tmp_path).paper_rag_toolsets == "none"
