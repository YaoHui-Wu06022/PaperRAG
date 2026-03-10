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

# 这个模块屏蔽本地 FAISS 和远端 Milvus/Zilliz 的差异，
# 让项目其它部分都可以统一按 doc_id 这一层来操作索引。

def _get_langchain_milvus_class():
    try:
        from langchain_milvus import Milvus

        return Milvus
    except ImportError:
        try:
            from langchain_community.vectorstores import Milvus

            return Milvus
        except ImportError as exc:
            raise ImportError(
                "Milvus backend requires langchain-milvus. "
                "Please `pip install langchain-milvus pymilvus milvus-lite`."
            ) from exc


def build_vector_index(
    backend: str,
    documents: list[Document],
    embeddings,
    persist_dir: Path,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
    milvus_papers_collection: str = "rag_pdf_papers",
    milvus_drop_old: bool = True,
    paper_documents: list[Document] | None = None,
    reference_documents: list[Document] | None = None,
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
):
    """从零开始构建一套新的向量索引。

    对 Milvus 来说，这一步最多会创建三层 collection：
    chunk-level 文档、paper-level 文档，以及可选的 references collection。
    """
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
        if paper_documents:
            _build_milvus_index(
                documents=paper_documents,
                embeddings=embeddings,
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                milvus_db_name=milvus_db_name,
                milvus_collection=milvus_papers_collection,
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
    milvus_papers_collection: str = "rag_pdf_papers",
    doc_ids_to_replace: list[str] | None = None,
    paper_documents: list[Document] | None = None,
    reference_documents: list[Document] | None = None,
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
):
    """按 doc_id 执行文档 upsert。

    Milvus 支持按 doc_id 先删后写，所以 ingest 可以保持增量；
    FAISS 不支持这一套，因此 FAISS 路径会退回全量重建。
    """
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
        _delete_milvus_by_doc_ids(
            uri,
            milvus_papers_collection,
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

    if paper_documents:
        _append_milvus_documents(
            documents=paper_documents,
            embeddings=embeddings,
            milvus_uri=uri,
            milvus_token=milvus_token,
            milvus_db_name=milvus_db_name,
            milvus_collection=milvus_papers_collection,
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
    milvus_papers_collection: str = "rag_pdf_papers",
    references_strategy: str = "keyword_only",
    milvus_references_collection: str = "rag_pdf_references",
) -> None:
    """从所有保存这篇论文的索引层里删除它。"""
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
    _delete_milvus_by_doc_ids(
        uri,
        milvus_papers_collection,
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
    """加载已有的向量索引或 collection，且不做任何写入。"""
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


def vector_index_entity_count(
    backend: str,
    persist_dir: Path,
    *,
    milvus_uri: str | None = None,
    milvus_token: str = "",
    milvus_db_name: str = "",
    milvus_collection: str = "rag_pdf_chunks",
) -> int | None:
    backend = backend.strip().lower()
    if backend == "faiss":
        if not _faiss_persist_dir(persist_dir).exists():
            return 0
        return None
    if backend == "milvus":
        return _milvus_collection_entity_count(
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
    """把 Milvus 的 cloud / lite 两种配置统一整理成同一种连接参数。"""
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
    # `langchain_milvus` 和较旧的 community wrapper 构造参数不完全一致，
    # 所以这里集中做一层兼容适配。
    Milvus = _get_langchain_milvus_class()

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

    Milvus = _get_langchain_milvus_class()

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
    Milvus = _get_langchain_milvus_class()

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


def _milvus_collection_entity_count(
    milvus_uri: str | None,
    collection_name: str,
    *,
    milvus_token: str = "",
    milvus_db_name: str = "",
) -> int:
    try:
        from pymilvus import Collection, connections, utility
    except ImportError as exc:
        raise ImportError(
            "Milvus collection checks require pymilvus. Please `pip install pymilvus`."
        ) from exc

    alias = f"rag_count_{abs(hash((milvus_uri, collection_name))) % 100000}"
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
            return 0
        collection = Collection(name=collection_name, using=alias)
        collection.load()
        return int(collection.num_entities)
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
