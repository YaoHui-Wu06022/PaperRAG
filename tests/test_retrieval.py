from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_rag.cli.main import build_parser
from paper_rag.config import Settings
from paper_rag.retrieval.data.chunks import load_chunk_documents
from paper_rag.retrieval.dense.cache import CachedEmbedder, EmbeddingCache
from paper_rag.retrieval.dense.embedding import EmbeddingClient
from paper_rag.retrieval.dense.milvus_store import MilvusStore
from paper_rag.retrieval.dense.service import run_index, run_search


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.dropped = False
        self.created: dict | None = None
        self.inserted: list[dict] = []
        self.loaded = False

    def has_collection(self, collection_name: str) -> bool:
        return True

    def drop_collection(self, collection_name: str) -> None:
        self.dropped = True

    def create_collection(self, **kwargs) -> None:
        self.created = kwargs

    def insert(self, collection_name: str, data: list[dict]) -> None:
        self.inserted.extend(data)

    def load_collection(self, collection_name: str) -> None:
        self.loaded = True

    def search(self, **kwargs):
        return [[{
            "distance": 0.88,
            "entity": {
                "chunk_id": "paper::chunk_0000",
                "paper_id": "paper",
                "chunk_index": 0,
                "title": "Paper",
                "section_path_text": "Abstract",
                "pages_text": "1",
                "text": "A useful result.",
            },
        }]]


class RetrievalTests(unittest.TestCase):
    def test_settings_reads_milvus_and_embedding_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join([
                    "MILVUS_URI=https://example.zillizcloud.com",
                    "MILVUS_TOKEN=token",
                    "MILVUS_DB_NAME=db",
                    "MILVUS_COLLECTION=chunks",
                    "EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "EMBEDDING_API_KEY=key",
                    "EMBEDDING_MODEL=text-embedding-v4",
                    "EMBEDDING_DIM=1024",
                    "EMBEDDING_BATCH_SIZE=7",
                    "EMBEDDING_CACHE_PATH=data/index/cache.jsonl",
                    "BAIDU_TRANSLATE_APP_ID=app",
                    "BAIDU_TRANSLATE_SECRET_KEY=secret",
                    "BAIDU_TRANSLATE_ENDPOINT=https://example.com/translate",
                    "BAIDU_TRANSLATE_DOMAIN=academic",
                    "PLAN_DENSE_TOP_K=11",
                    "PLAN_BM25_TOP_K=12",
                    "PLAN_FINAL_TOP_K=4",
                    "PLAN_BLOCK_WINDOW=3",
                ]),
                encoding="utf-8",
            )
            settings = Settings.load(root)
            self.assertEqual(settings.milvus_uri, "https://example.zillizcloud.com")
            self.assertEqual(settings.milvus_token, "token")
            self.assertEqual(settings.milvus_db_name, "db")
            self.assertEqual(settings.milvus_collection, "chunks")
            self.assertEqual(settings.embedding_model, "text-embedding-v4")
            self.assertEqual(settings.embedding_dim, 1024)
            self.assertEqual(settings.embedding_batch_size, 7)
            self.assertEqual(settings.embedding_cache_path, root / "data" / "index" / "cache.jsonl")
            self.assertEqual(settings.baidu_translate_app_id, "app")
            self.assertEqual(settings.baidu_translate_secret_key, "secret")
            self.assertEqual(settings.baidu_translate_endpoint, "https://example.com/translate")
            self.assertEqual(settings.baidu_translate_domain, "academic")
            self.assertEqual(settings.plan_dense_top_k, 11)
            self.assertEqual(settings.plan_bm25_top_k, 12)
            self.assertEqual(settings.plan_final_top_k, 4)
            self.assertEqual(settings.plan_block_window, 3)

    def test_embedding_client_uses_openai_compatible_payload(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        client = EmbeddingClient(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "secret",
            "text-embedding-v4",
            1024,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            vectors = client.embed_texts(["hello"])
        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(captured["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["model"], "text-embedding-v4")
        self.assertEqual(captured["payload"]["input"], ["hello"])
        self.assertEqual(captured["payload"]["dimensions"], 1024)

    def test_embedding_cache_hits_skip_client_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "cache.jsonl")
            client = FakeEmbeddingClient()
            embedder = CachedEmbedder(client, cache, model="m", dimensions=2, batch_size=10)
            first = embedder.embed_texts(["same"])
            second = embedder.embed_texts(["same"])
            self.assertEqual(first, second)
            self.assertEqual(len(client.calls), 1)

    def test_chunk_loader_reads_chunks_with_metadata_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "paper_data" / "Paper_abc"
            paper.mkdir(parents=True)
            (paper / "metadata.json").write_text(json.dumps({"title": "Paper Title"}), encoding="utf-8")
            (paper / "chunks.jsonl").write_text(
                json.dumps({
                    "chunk_id": "Paper_abc::chunk_0000",
                    "paper_id": "Paper_abc",
                    "chunk_index": 0,
                    "section_path": ["Abstract"],
                    "pages": [1],
                    "text": "Body",
                    "embedding_text": "Paper: Paper Title\n\nBody",
                }) + "\n",
                encoding="utf-8",
            )
            docs = load_chunk_documents(Path(tmp) / "paper_data")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].title, "Paper Title")
            self.assertEqual(docs[0].section_path_text, "Abstract")

    def test_milvus_store_recreates_collection_and_maps_search(self) -> None:
        client = FakeMilvusClient()
        store = MilvusStore(
            uri="x",
            token="t",
            db_name="",
            collection_name="chunks",
            dimensions=2,
            client=client,
        )
        store.recreate_collection()
        self.assertTrue(client.dropped)
        self.assertEqual(client.created["collection_name"], "chunks")
        self.assertEqual(client.created["dimension"], 2)
        self.assertEqual(client.created["metric_type"], "COSINE")
        results = store.search([0.1, 0.2], top_k=1)
        self.assertEqual(results[0].chunk_id, "paper::chunk_0000")
        self.assertEqual(results[0].score, 0.88)

    def test_index_and_search_services_use_chunks_not_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "data" / "paper_data" / "Paper_abc"
            paper.mkdir(parents=True)
            (paper / "metadata.json").write_text(json.dumps({"title": "Paper Title"}), encoding="utf-8")
            (paper / "chunks.jsonl").write_text(
                json.dumps({
                    "chunk_id": "Paper_abc::chunk_0000",
                    "paper_id": "Paper_abc",
                    "chunk_index": 0,
                    "section_path": ["Abstract"],
                    "pages": [1],
                    "text": "Body",
                    "embedding_text": "Paper: Paper Title\n\nBody",
                }) + "\n",
                encoding="utf-8",
            )
            settings = Settings.load(root)
            embedder = FakeEmbeddingClient()
            fake_client = FakeMilvusClient()
            store = MilvusStore(
                uri="x",
                token=None,
                db_name=None,
                collection_name="chunks",
                dimensions=2,
                client=fake_client,
            )
            summary = run_index(settings, reporter=lambda _: None, embedder=embedder, store=store)
            self.assertEqual(summary.chunk_count, 1)
            self.assertEqual(fake_client.inserted[0]["title"], "Paper Title")
            results = run_search(settings, "query", top_k=1, embedder=embedder, store=store)
            self.assertEqual(results[0].snippet, "A useful result.")

    def test_cli_knows_index_and_search(self) -> None:
        parser = build_parser()
        index_args = parser.parse_args(["index", "--quiet"])
        search_args = parser.parse_args(["search", "center loss", "--top-k", "3"])
        plan_args = parser.parse_args(["plan", "center loss 如何提升类内紧凑性"])
        ask_args = parser.parse_args(["ask", "center loss"])
        self.assertEqual(index_args.command, "index")
        self.assertEqual(search_args.command, "search")
        self.assertEqual(search_args.top_k, 3)
        self.assertEqual(plan_args.command, "plan")
        self.assertEqual(ask_args.command, "ask")


if __name__ == "__main__":
    unittest.main()
