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

    def query(self, query_texts: List[str], n_results: int = 5) -> List[List[str]]:
        raise NotImplementedError

@storage_registry.register("chromadb")
class ChromaDBStore(VectorStore):
    def __init__(self, persist_directory: str = ".chroma_db", collection_name: str = "digital_clone"):
        if not HAS_CHROMA:
            raise ImportError("chromadb is required to use ChromaDBStore. Run: pip install chromadb")
            
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDBStore initialized at {persist_directory}, collection: {collection_name}")

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        if not documents:
            return
            
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_texts: List[str], n_results: int = 5) -> List[List[str]]:
        if not query_texts:
            return []
            
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        # ChromaDB returns a list of lists of documents
        return results.get("documents", [])
