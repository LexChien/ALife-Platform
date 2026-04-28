import logging
import uuid
from typing import List

from core.storage import storage_registry

logger = logging.getLogger(__name__)

class MemoryStore:
    def __init__(self, collection_name="digital_clone", use_vector_db=True, persist_directory=".chroma_db"):
        self.items = []
        self.vector_store = None
        self.use_vector_db = False
        if not use_vector_db:
            return
        try:
            self.vector_store = storage_registry.create(
                "chromadb",
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
            self.use_vector_db = True
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDBStore: {e}. Falling back to in-memory list only.")
            self.vector_store = None
            self.use_vector_db = False

    def add(self, role, content, kind="dialogue"):
        item = {"role": role, "content": content, "kind": kind}
        self.items.append(item)
        
        if self.use_vector_db:
            try:
                self.vector_store.add_documents(
                    documents=[content],
                    metadatas=[{"role": role, "kind": kind}],
                    ids=[str(uuid.uuid4())]
                )
            except Exception as e:
                logger.error(f"Failed to add document to vector store: {e}")

    def recent(self, n=5):
        return self.items[-n:]

    def add_profile_facts(self, facts):
        for fact in facts:
            self.add("system", fact, kind="profile_fact")

    def retrieve(self, query, n=3, kinds=None):
        if not self.items: 
            return []
            
        if self.use_vector_db:
            try:
                docs = self.vector_store.query([query], n_results=n * 3)
                retrieved = []
                seen_contents = set()
                
                if docs and len(docs) > 0:
                    for doc_content in docs[0]:
                        if doc_content in seen_contents:
                            continue
                        # Find the original item in self.items
                        matched_items = [i for i in self.items if i["content"] == doc_content]
                        for item in matched_items:
                            if kinds is None or item["kind"] in kinds:
                                retrieved.append(item)
                                seen_contents.add(doc_content)
                                break
                        if len(retrieved) >= n:
                            break
                return retrieved
            except Exception as e:
                logger.error(f"Vector search failed: {e}. Falling back to recent.")
                
        # Fallback if no vector db or error
        filtered = [i for i in self.items if kinds is None or i["kind"] in kinds]
        return filtered[-n:]

    def retrieve_for_prompt(self, user_text, limit=5):
        profile = self.retrieve(user_text, n=max(limit // 2, 1), kinds=["profile_fact"])
        dialogue = self.retrieve(user_text, n=limit, kinds=["dialogue"])
        merged = []
        seen = set()
        for item in profile + dialogue:
            content = item.get("content")
            role = item.get("role")
            if not content or content == user_text or content in seen:
                continue
            if item.get("kind") == "dialogue" and role != "user":
                continue
            merged.append(item)
            seen.add(content)
            if len(merged) >= limit:
                break
        return merged
