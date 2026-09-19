"""Print a redacted embedding service error for troubleshooting."""

from pathlib import Path
from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import Embedder

settings = LibrarySettings.load(Path.cwd())
with Catalog(settings) as cat:
    try:
        vectors = Embedder(cat).embed(
            ["How do residual connections affect Inception training speed?"]
        )
        print("Embedding succeeded", len(vectors[0]))
    except Exception as exc:
        message = str(exc)
        for key, value in settings.env.items():
            if value and any(
                word in key.upper() for word in ("KEY", "TOKEN", "SECRET")
            ):
                message = message.replace(value, "[REDACTED]")
        print(type(exc).__name__, message)
