from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
import warnings

from langchain_core.documents import Document


warnings.filterwarnings(
    "ignore",
    message=r"The class `Milvus` was deprecated in LangChain 0\.2\.0.*",
)
FAISS_INDEX_DIRNAME = "faiss_index"


def build_vector_index(
    backend: str,
    documents: list[Document],
    embeddings,
    persist_dir: Path,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
    milvus_drop_old: bool = True,
    reference_documents: list[Document] | None = None,
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
):
    backend = backend.strip().lower()
    if backend == "faiss":
        return _build_faiss_index(documents, embeddings, persist_dir)

    if backend == "milvus":
        store = _build_milvus_index(
            documents=documents,
            embeddings=embeddings,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_collection,
            milvus_drop_old=milvus_drop_old,
        )
        if (
            references_strategy.strip().lower() == "separate_collection"
            and reference_documents
        ):
            _build_milvus_index(
                documents=reference_documents,
                embeddings=embeddings,
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                milvus_db_name=milvus_db_name,
                milvus_collection=milvus_references_collection,
                milvus_drop_old=milvus_drop_old,
            )
        return store

    raise ValueError(f"Unsupported vector backend: {backend}. Use 'faiss' or 'milvus'.")


def upsert_vector_index(
    backend: str,
    documents: list[Document],
    embeddings,
    persist_dir: Path,
    *,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
    doc_ids_to_replace: list[str] | None = None,
    reference_documents: list[Document] | None = None,
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
):
    backend = backend.strip().lower()
    replace_ids = sorted({str(item).strip() for item in (doc_ids_to_replace or []) if str(item).strip()})

    if backend == "faiss":
        # FAISS 不支持按 doc_id 删除，增量时回退为全量重建（documents 需为全量语料）。
        return _build_faiss_index(documents, embeddings, persist_dir)

    if backend != "milvus":
        raise ValueError(f"Unsupported vector backend: {backend}. Use 'faiss' or 'milvus'.")

    uri = _normalize_milvus_uri(milvus_uri)
    if replace_ids:
        _delete_milvus_by_doc_ids(
            uri,
            milvus_collection,
            replace_ids,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
        )
    if documents:
        store = _append_milvus_documents(
            documents=documents,
            embeddings=embeddings,
            milvus_uri=uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_collection,
        )
    else:
        store = _load_milvus_index(
            embeddings=embeddings,
            milvus_uri=uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_collection,
        )

    if references_strategy.strip().lower() == "separate_collection":
        if replace_ids:
            _delete_milvus_by_doc_ids(
                uri,
                milvus_references_collection,
                replace_ids,
                milvus_token=milvus_token,
                milvus_db_name=milvus_db_name,
            )
        if reference_documents:
            _append_milvus_documents(
                documents=reference_documents,
                embeddings=embeddings,
                milvus_uri=uri,
                milvus_token=milvus_token,
                milvus_db_name=milvus_db_name,
                milvus_collection=milvus_references_collection,
            )
    return store


def delete_documents_from_index(
    backend: str,
    *,
    embeddings,
    persist_dir: Path,
    doc_ids: list[str],
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
) -> None:
    backend = backend.strip().lower()
    normalized_ids = sorted({str(item).strip() for item in doc_ids if str(item).strip()})
    if not normalized_ids:
        return

    if backend == "faiss":
        # FAISS 场景下由上层通过本地语料重建索引。
        return

    if backend != "milvus":
        raise ValueError(f"Unsupported vector backend: {backend}. Use 'faiss' or 'milvus'.")

    uri = _normalize_milvus_uri(milvus_uri)
    _delete_milvus_by_doc_ids(
        uri,
        milvus_collection,
        normalized_ids,
        milvus_token=milvus_token,
        milvus_db_name=milvus_db_name,
    )
    if references_strategy.strip().lower() == "separate_collection":
        _delete_milvus_by_doc_ids(
            uri,
            milvus_references_collection,
            normalized_ids,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
        )


def load_vector_index(
    backend: str,
    embeddings,
    persist_dir: Path,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
):
    backend = backend.strip().lower()
    if backend == "faiss":
        return _load_faiss_index(embeddings, persist_dir)

    if backend == "milvus":
        return _load_milvus_index(
            embeddings=embeddings,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_collection,
        )

    raise ValueError(f"Unsupported vector backend: {backend}. Use 'faiss' or 'milvus'.")


def vector_index_exists(
    backend: str,
    persist_dir: Path,
    *,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
) -> bool:
    backend = backend.strip().lower()
    if backend == "faiss":
        return _faiss_persist_dir(persist_dir).exists()

    if backend == "milvus":
        return _milvus_collection_exists(
            milvus_uri,
            milvus_collection,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
        )

    raise ValueError(f"Unsupported vector backend: {backend}. Use 'faiss' or 'milvus'.")


def _build_faiss_index(documents: list[Document], embeddings, persist_dir: Path):
    from langchain_community.vectorstores import FAISS

    faiss_dir = _faiss_persist_dir(persist_dir)
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(faiss_dir))
    return store


def _load_faiss_index(embeddings, persist_dir: Path):
    from langchain_community.vectorstores import FAISS

    faiss_dir = _faiss_persist_dir(persist_dir)
    if not faiss_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found: {faiss_dir}. Please run ingestion first."
        )
    return FAISS.load_local(
        str(faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _faiss_persist_dir(persist_dir: Path) -> Path:
    return persist_dir / FAISS_INDEX_DIRNAME


def _normalize_milvus_uri(milvus_uri: str | None) -> str:
    if milvus_uri is None or not milvus_uri.strip():
        raise ValueError(
            "Milvus backend requires MILVUS_URI. "
            "Set it in .env (e.g., your Zilliz Cloud endpoint or http://localhost:19530)."
        )
    uri = milvus_uri.strip()
    parsed = urlparse(uri)

    if not parsed.scheme:
        uri_path = Path(uri).resolve()
        uri_path.parent.mkdir(parents=True, exist_ok=True)
        return str(uri_path)
    return uri


def build_milvus_connection_args(
    milvus_uri: str | None,
    *,
    milvus_token: str = "",
    milvus_db_name: str = "",
) -> dict[str, str]:
    connection_args = {"uri": _normalize_milvus_uri(milvus_uri)}
    token = str(milvus_token or "").strip()
    db_name = str(milvus_db_name or "").strip()
    if token:
        connection_args["token"] = token
    if db_name:
        connection_args["db_name"] = db_name
    return connection_args


def _build_milvus_index(
    documents: list[Document],
    embeddings,
    milvus_uri: str | None,
    milvus_token: str,
    milvus_db_name: str,
    milvus_collection: str,
    milvus_drop_old: bool,
):
    try:
        from langchain_community.vectorstores import Milvus
    except ImportError as exc:
        raise ImportError(
            "Milvus backend requires pymilvus. Please `pip install pymilvus milvus-lite`."
        ) from exc

    connection_args = build_milvus_connection_args(
        milvus_uri,
        milvus_token=milvus_token,
        milvus_db_name=milvus_db_name,
    )
    docs_for_milvus = _to_milvus_documents(documents)

    try:
        return Milvus.from_documents(
            documents=docs_for_milvus,
            embedding=embeddings,
            collection_name=milvus_collection,
            connection_args=connection_args,
            drop_old=milvus_drop_old,
        )
    except TypeError:
        return Milvus.from_documents(
            docs_for_milvus,
            embeddings,
            collection_name=milvus_collection,
            connection_args=connection_args,
            drop_old=milvus_drop_old,
        )


def _append_milvus_documents(
    documents: list[Document],
    embeddings,
    milvus_uri: str,
    milvus_token: str,
    milvus_db_name: str,
    milvus_collection: str,
):
    if not documents:
        return _load_milvus_index(
            embeddings=embeddings,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_collection,
        )

    try:
        from langchain_community.vectorstores import Milvus
    except ImportError as exc:
        raise ImportError(
            "Milvus backend requires pymilvus. Please `pip install pymilvus milvus-lite`."
        ) from exc

    connection_args = build_milvus_connection_args(
        milvus_uri,
        milvus_token=milvus_token,
        milvus_db_name=milvus_db_name,
    )
    docs_for_milvus = _to_milvus_documents(documents)

    try:
        return Milvus.from_documents(
            documents=docs_for_milvus,
            embedding=embeddings,
            collection_name=milvus_collection,
            connection_args=connection_args,
            drop_old=False,
        )
    except TypeError:
        return Milvus.from_documents(
            docs_for_milvus,
            embeddings,
            collection_name=milvus_collection,
            connection_args=connection_args,
            drop_old=False,
        )


def _load_milvus_index(
    embeddings,
    milvus_uri: str | None,
    milvus_token: str,
    milvus_db_name: str,
    milvus_collection: str,
):
    try:
        from langchain_community.vectorstores import Milvus
    except ImportError as exc:
        raise ImportError(
            "Milvus backend requires pymilvus. Please `pip install pymilvus milvus-lite`."
        ) from exc

    connection_args = build_milvus_connection_args(
        milvus_uri,
        milvus_token=milvus_token,
        milvus_db_name=milvus_db_name,
    )

    try:
        return Milvus(
            embedding_function=embeddings,
            collection_name=milvus_collection,
            connection_args=connection_args,
        )
    except TypeError:
        try:
            return Milvus(
                embedding=embeddings,
                collection_name=milvus_collection,
                connection_args=connection_args,
            )
        except TypeError:
            try:
                return Milvus.from_existing_collection(
                    embedding=embeddings,
                    collection_name=milvus_collection,
                    connection_args=connection_args,
                )
            except (TypeError, AttributeError):
                return Milvus.from_existing_collection(
                    embedding_function=embeddings,
                    collection_name=milvus_collection,
                    connection_args=connection_args,
                )


def _delete_milvus_by_doc_ids(
    milvus_uri: str,
    collection_name: str,
    doc_ids: list[str],
    *,
    milvus_token: str = "",
    milvus_db_name: str = "",
) -> None:
    if not doc_ids:
        return
    try:
        from pymilvus import Collection, connections, utility
    except ImportError as exc:
        raise ImportError(
            "Milvus delete requires pymilvus. Please `pip install pymilvus`."
        ) from exc

    alias = f"rag_delete_{abs(hash((milvus_uri, collection_name))) % 100000}"
    connections.connect(
        alias=alias,
        **build_milvus_connection_args(
            milvus_uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
        ),
    )
    try:
        if not utility.has_collection(collection_name=collection_name, using=alias):
            return
        collection = Collection(name=collection_name, using=alias)
        batch_size = 200
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            quoted = ", ".join(_quote_milvus_string(item) for item in batch)
            expr = f"doc_id in [{quoted}]"
            collection.delete(expr=expr)
        collection.flush()
    finally:
        connections.disconnect(alias=alias)


def _milvus_collection_exists(
    milvus_uri: str | None,
    collection_name: str,
    *,
    milvus_token: str = "",
    milvus_db_name: str = "",
) -> bool:
    try:
        from pymilvus import connections, utility
    except ImportError as exc:
        raise ImportError(
            "Milvus collection checks require pymilvus. Please `pip install pymilvus`."
        ) from exc

    alias = f"rag_exists_{abs(hash((milvus_uri, collection_name))) % 100000}"
    connections.connect(
        alias=alias,
        **build_milvus_connection_args(
            milvus_uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
        ),
    )
    try:
        return bool(utility.has_collection(collection_name=collection_name, using=alias))
    finally:
        connections.disconnect(alias=alias)


def _quote_milvus_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _to_milvus_documents(documents: list[Document]) -> list[Document]:
    safe_docs: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        normalized = {
            str(key): _normalize_milvus_metadata_value(value)
            for key, value in metadata.items()
        }
        safe_docs.append(Document(page_content=doc.page_content, metadata=normalized))
    return safe_docs


def _normalize_milvus_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)
