"""The vector index must be rebuilt when the embedding model changes.

Before Nov 2026 the retriever reused an existing ChromaDB collection whenever
the number of chunks matched. Switching the embedding model keeps the chunk
count, so the old (incompatible) vectors would have been reused silently.
"""
import sys
import types

import numpy as np

from src.rag.retriever import ClinicalRetriever


class _FakeCollection:
    def __init__(self, count, metadata):
        self._count = count
        self.metadata = metadata
        self.added = 0

    def count(self):
        return self._count

    def add(self, ids, embeddings, documents, metadatas):
        self.added += len(ids)


class _FakeClient:
    def __init__(self, existing):
        self.existing = existing
        self.deleted = False
        self.created = None

    def get_collection(self, name):
        if self.existing is None:
            raise ValueError("no collection")
        return self.existing

    def delete_collection(self, name):
        self.deleted = True

    def create_collection(self, name, metadata):
        self.created = _FakeCollection(0, metadata)
        return self.created


def _run(monkeypatch, existing, model="model-B"):
    client = _FakeClient(existing)
    fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path, settings: client)
    fake_config = types.SimpleNamespace(Settings=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.config", fake_config)

    r = ClinicalRetriever.__new__(ClinicalRetriever)
    r.chromadb_path = "unused"
    r.collection_name = "hematology_knowledge"
    r.embedding_model_name = model
    r.chunks = [{"text": f"t{i}", "source": "a.pdf", "chunk_id": i, "total_chunks": 3} for i in range(3)]
    r.embeddings = np.zeros((3, 4))
    r._init_chromadb()
    return r, client


def test_reuses_index_built_with_same_model(monkeypatch):
    existing = _FakeCollection(3, {"embedding_model": "model-B"})
    r, client = _run(monkeypatch, existing)
    assert r.chroma_collection is existing
    assert not client.deleted


def test_rebuilds_when_model_changed_even_if_count_matches(monkeypatch):
    existing = _FakeCollection(3, {"embedding_model": "model-A"})
    r, client = _run(monkeypatch, existing)
    assert client.deleted
    assert client.created.metadata["embedding_model"] == "model-B"
    assert client.created.added == 3


def test_rebuilds_old_index_without_model_tag(monkeypatch):
    existing = _FakeCollection(3, {"hnsw:space": "cosine"})
    r, client = _run(monkeypatch, existing)
    assert client.deleted
    assert client.created.added == 3
