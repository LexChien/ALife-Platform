import logging
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from core.registry import Registry

logger = logging.getLogger(__name__)

storage_registry = Registry()

class VectorStore:
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        raise NotImplementedError

    def query(self, query_texts: List[str], n_results: int = 5, include_metadata: bool = False, where=None):
        raise NotImplementedError

@storage_registry.register("chromadb")
class ChromaDBStore(VectorStore):
    def __init__(self, persist_directory: str = ".chroma_db", collection_name: str = "digital_clone", embedding_function=None):
        if not HAS_CHROMA:
            raise ImportError("chromadb is required to use ChromaDBStore. Run: pip install chromadb")
            
        self.client = chromadb.PersistentClient(path=persist_directory)
        kwargs = {"name": collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(**kwargs)
        logger.info(f"ChromaDBStore initialized at {persist_directory}, collection: {collection_name}")

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        if not documents:
            return
            
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_texts: List[str], n_results: int = 5, include_metadata: bool = False, where=None):
        if not query_texts or n_results <= 0:
            return []
        available = self.collection.count()
        if not available:
            return [[] for _ in query_texts]
        kwargs = {"include": ["documents", "metadatas"] if include_metadata else ["documents"]}
        if where is not None:
            kwargs["where"] = where
        results = self.collection.query(
            query_texts=query_texts,
            n_results=min(n_results, available),
            **kwargs,
        )
        if not include_metadata:
            return results.get("documents", []) or []
            
        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []
        
        ret = []
        for i in range(len(documents)):
            sub = []
            for j in range(len(documents[i] or [])):
                doc_text = documents[i][j]
                meta = metadatas[i][j] if len(metadatas) > i and metadatas[i] and len(metadatas[i]) > j else {}
                doc_id = ids[i][j] if len(ids) > i and ids[i] and len(ids[i]) > j else None
                sub.append({
                    "document": doc_text,
                    "metadata": meta or {},
                    "id": doc_id
                })
            ret.append(sub)
        return ret
