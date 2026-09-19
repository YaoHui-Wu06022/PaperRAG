import json
from pathlib import Path
from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings


def main():
    with Catalog(LibrarySettings.load(Path.cwd())) as cat:
        print("KINDS", cat.rows("SELECT kind,count(*) n FROM blocks GROUP BY kind"))
        for selector in ("Attention is All", "Adam:", "BERT:"):
            doc = next(
                d
                for d in cat.rows("SELECT * FROM documents")
                if json.loads(d["metadata"])["title"].startswith(selector)
            )
            print("DOCUMENT", json.loads(doc["metadata"])["title"], doc["document_id"])
            for kind in ("equation_interline", "table", "appendix"):
                condition = (
                    "region='appendix' AND length(text)>100"
                    if kind == "appendix"
                    else "kind=?"
                )
                args = [doc["active_revision"]] + ([] if kind == "appendix" else [kind])
                rows = cat.rows(
                    "SELECT block_id,region,kind,text,section_id FROM blocks WHERE revision_id=? AND "
                    + condition
                    + " ORDER BY ordinal LIMIT 2",
                    args,
                )
                print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
