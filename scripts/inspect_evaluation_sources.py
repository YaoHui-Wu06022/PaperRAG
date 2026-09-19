"""Print candidate primary source spans for host-agent benchmark review."""

import json
from pathlib import Path
from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.common import evidence_id


def main():
    with Catalog(LibrarySettings.load(Path.cwd())) as cat:
        for doc in cat.rows("SELECT * FROM documents ORDER BY document_id"):
            rows = cat.rows(
                "SELECT * FROM blocks WHERE revision_id=? AND region='abstract' AND length(text)>100 ORDER BY ordinal",
                (doc["active_revision"],),
            )
            if not rows:
                rows = cat.rows(
                    "SELECT * FROM blocks WHERE revision_id=? AND region='body' AND length(text)>250 ORDER BY ordinal LIMIT 1",
                    (doc["active_revision"],),
                )
            print(
                json.dumps(
                    {
                        "title": json.loads(doc["metadata"])["title"],
                        "document_id": doc["document_id"],
                        "spans": [
                            {
                                "id": evidence_id(
                                    doc["document_id"],
                                    r["revision_id"],
                                    r["block_id"],
                                    0,
                                    len(r["text"]),
                                ),
                                "text": r["text"],
                            }
                            for r in rows
                        ],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
