"""Build the independent v2 collection without touching legacy vectors."""

import json
import sys
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import index_library


def main():
    with Catalog(LibrarySettings.load(Path.cwd()), writable=True) as cat:
        result = index_library(
            cat, reporter=lambda message: print(message, file=sys.stderr, flush=True)
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
