"""Multilingual (zh/en) embedding functions for DigiClone memory (Plan 37 R2).

Benchmark (runs/plan37/zh_retrieval/20261002-062625, 30 zh-TW fact/paraphrase pairs, real models):
  chroma default all-MiniLM-L6-v2  R@1 0.167  R@3 0.233
  BAAI/bge-small-zh-v1.5            R@1 0.733  R@3 0.933
  intfloat/multilingual-e5-small    R@1 0.833  R@3 0.967   <- default here (zh + en)
E5 models expect "query: " / "passage: " prefixes; Chroma calls embed_query() for queries.
"""
from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"
# Chroma re-instantiates the EF via build_from_config() on collection operations; cache the
# loaded SentenceTransformer per (model, device) so that is cheap (R2: 28 reloads in 30 adds).
_MODEL_CACHE: Dict[tuple, Any] = {}

try:
    from chromadb import Documents, EmbeddingFunction, Embeddings
except Exception:  # pragma: no cover - chroma missing: MemoryStore already falls back
    EmbeddingFunction = object  # type: ignore
    Documents = Embeddings = List  # type: ignore


class PrefixedSentenceTransformerEF(EmbeddingFunction):  # type: ignore[misc]
    """SentenceTransformer EF with separate document/query prefixes (E5-style)."""

    def __init__(self, model_name: str = DEFAULT_MULTILINGUAL_MODEL, device: str = "cpu",
                 doc_prefix: str | None = None, query_prefix: str | None = None):
        self.model_name = model_name
        self.device = device
        is_e5 = "e5" in model_name.lower()
        self.doc_prefix = doc_prefix if doc_prefix is not None else ("passage: " if is_e5 else "")
        self.query_prefix = query_prefix if query_prefix is not None else ("query: " if is_e5 else "")
        key = (model_name, device)
        if key not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
        self._model = _MODEL_CACHE[key]

    def _encode(self, texts: List[str]) -> Embeddings:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return [v.astype("float32") for v in vecs]

    def __call__(self, input: Documents) -> Embeddings:  # documents
        return self._encode([self.doc_prefix + t for t in input])

    def embed_query(self, input: Documents) -> Embeddings:
        return self._encode([self.query_prefix + t for t in input])

    @staticmethod
    def name() -> str:
        return "alife_prefixed_sentence_transformer"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "device": self.device,
                "doc_prefix": self.doc_prefix, "query_prefix": self.query_prefix}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "PrefixedSentenceTransformerEF":
        return PrefixedSentenceTransformerEF(**config)

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]


def collection_suffix(model_name: str) -> str:
    """Collections are tied to one embedding space; derive a stable suffix per model."""
    import re
    return "__" + re.sub(r"[^a-z0-9]+", "_", model_name.lower().split("/")[-1]).strip("_")


def migrate_collection(persist_directory: str, source_name: str, target_store) -> int:
    """Copy documents+metadata+ids from an old (default-embedding) collection into target_store.

    Returns number of copied items; 0 if the source does not exist or target already has data.
    The source collection is left untouched (no deletion).
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(persist_directory))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if source_name not in names:
        return 0
    if target_store.collection.count() > 0:
        return 0
    src = client.get_collection(source_name)
    data = src.get(include=["documents", "metadatas"])
    docs, metas, ids = data.get("documents") or [], data.get("metadatas") or [], data.get("ids") or []
    if not docs:
        return 0
    target_store.add_documents(documents=list(docs), metadatas=[m or {} for m in metas], ids=list(ids))
    return len(docs)
