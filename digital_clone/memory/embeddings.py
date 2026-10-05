"""Multilingual (zh/en) embedding functions for DigiClone memory (Plan 37 R2).

Benchmark (runs/plan37/zh_retrieval/20261002-062625, 30 zh-TW fact/paraphrase pairs, real models):
  chroma default all-MiniLM-L6-v2  R@1 0.167  R@3 0.233
  BAAI/bge-small-zh-v1.5            R@1 0.733  R@3 0.933
  intfloat/multilingual-e5-small    R@1 0.833  R@3 0.967   <- default here (zh + en)
E5 models expect "query: " / "passage: " prefixes; Chroma calls embed_query() for queries.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

DEFAULT_MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"
# Chroma re-instantiates the EF via build_from_config() on collection operations; cache the
# loaded SentenceTransformer per (model, device) so that is cheap (R2: 28 reloads in 30 adds).
_MODEL_CACHE: Dict[tuple, Any] = {}
# 2026-10-05: torch MPS MetalShaderLibrary is not multi-thread safe (gemma_web SIGSEGV when two
# voice_chat turns encoded with shared bge-m3 concurrently). Serialise every encode on the shared model.
_ENCODE_LOCK = threading.RLock()
_CACHE_LOCK = threading.Lock()

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
        with _CACHE_LOCK:
            if key not in _MODEL_CACHE:
                from sentence_transformers import SentenceTransformer
                _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
            self._model = _MODEL_CACHE[key]

    def _encode(self, texts: List[str]) -> Embeddings:
        # Hold the encode lock for the whole MPS forward; concurrent HTTP/pool threads must queue.
        with _ENCODE_LOCK:
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


def migrate_collection(persist_directory: str, source_name, target_store) -> int:
    """Copy documents+metadata+ids from older collection(s) into target_store (one-time, non-destructive).

    source_name: a collection name, or a list of names. R2: when switching embeddings again (e5 -> bge-m3) the
    union of all listed sources is copied, de-duplicated by id (first source listed wins), so memories written
    under an intermediate embedding are not lost. Returns copied count; 0 if target already has data.
    Source collections are never deleted.
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(persist_directory))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    sources = [source_name] if isinstance(source_name, str) else list(source_name)
    target_name = getattr(target_store.collection, "name", None)
    sources = [n for n in sources if n in names and n != target_name]
    if not sources or target_store.collection.count() > 0:
        return 0
    seen, docs, metas, ids = set(), [], [], []
    for name in sources:
        data = client.get_collection(name).get(include=["documents", "metadatas"])
        for doc, meta, mid in zip(data.get("documents") or [], data.get("metadatas") or [], data.get("ids") or []):
            if mid in seen or not doc:
                continue
            seen.add(mid)
            docs.append(doc)
            metas.append(meta or {})
            ids.append(mid)
    if not docs:
        return 0
    target_store.add_documents(documents=docs, metadatas=metas, ids=ids)
    return len(docs)


def migration_sources(persist_directory: str, base_collection: str, target_collection: str) -> list:
    """Prior collections for this base: embedding-suffixed ones (largest first), then the original base."""
    import chromadb
    client = chromadb.PersistentClient(path=str(persist_directory))
    cols = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    suffixed = [n for n in cols if n.startswith(base_collection + "__") and n != target_collection]
    suffixed.sort(key=lambda n: client.get_collection(n).count(), reverse=True)
    return suffixed + ([base_collection] if base_collection in cols else [])
