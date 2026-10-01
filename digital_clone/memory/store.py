import logging
import uuid

from core.storage import storage_registry

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, collection_name="digital_clone", persist_directory=None,
                 require_persistence=False, embedding_function=None):
        self.items = []
        self.collection_name = collection_name
        self.require_persistence = require_persistence
        params = {"collection_name": collection_name}
        if persist_directory is not None:
            params["persist_directory"] = str(persist_directory)
        if embedding_function is not None:
            params["embedding_function"] = embedding_function
        try:
            self.vector_store = storage_registry.create("chromadb", **params)
            self.use_vector_db = True
        except Exception as exc:
            if require_persistence:
                raise RuntimeError("Persistent memory is required but ChromaDB is unavailable") from exc
            logger.warning("ChromaDB unavailable: %s. Using nonpersistent memory.", exc)
            self.vector_store = None
            self.use_vector_db = False

    def add(self, role, content, kind="dialogue", memory_id=None, scope=None):
        memory_id = memory_id or str(uuid.uuid4())
        item = {"id": memory_id, "role": role, "content": content, "kind": kind}
        metadata = {"role": role, "kind": kind}
        if scope is not None:
            item["scope"] = str(scope)
            metadata["scope"] = str(scope)
        if self.use_vector_db:
            try:
                self.vector_store.add_documents(
                    documents=[content], metadatas=[metadata],
                    ids=[memory_id],
                )
            except Exception as exc:
                if self.require_persistence:
                    raise RuntimeError("Persistent memory write failed") from exc
                logger.error("Persistent memory write failed: %s", exc)
        self.items = [old for old in self.items if old.get("id") != memory_id]
        self.items.append(item)
        return item

    def recent(self, n=5):
        return self.items[-n:] if n > 0 else []

    def add_profile_facts(self, facts):
        for fact in facts:
            identity = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:profile_fact:{fact}"))
            self.add("system", fact, kind="profile_fact", memory_id=identity)

    def retrieve(self, query, n=3, kinds=None, roles=None, scope=None):
        if n <= 0 or kinds == [] or roles == []:
            return []
        if self.use_vector_db:
            try:
                options = {"n_results": n * 3, "include_metadata": True}
                filters = []
                if kinds is not None:
                    filters.append({"kind": {"$in": list(kinds)}})
                if roles is not None:
                    filters.append({"role": {"$in": list(roles)}})
                if scope is not None:
                    filters.append({"scope": str(scope)})
                if filters:
                    options["where"] = filters[0] if len(filters) == 1 else {"$and": filters}
                docs = self.vector_store.query([query], **options)
                retrieved = []
                seen = set()
                for document in (docs[0] or []) if docs else []:
                    content = document.get("document")
                    metadata = document.get("metadata") or {}
                    role = metadata.get("role", "assistant")
                    kind = metadata.get("kind", "dialogue")
                    if not isinstance(content, str) or not content or (kinds is not None and kind not in kinds):
                        continue
                    if roles is not None and role not in roles:
                        continue
                    if scope is not None and metadata.get("scope") != str(scope):
                        continue
                    identity = document.get("id")
                    key = ("id", identity) if identity is not None else ("fields", role, kind, content)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Database metadata is authoritative, even when current-session text matches.
                    retrieved.append({"id": identity, "content": content, "role": role, "kind": kind})
                    if len(retrieved) >= n:
                        break
                return retrieved
            except Exception as exc:
                if self.require_persistence:
                    raise RuntimeError("Persistent memory retrieval failed") from exc
                logger.error("Vector retrieval failed: %s. Using current-session memory.", exc)
        filtered = [item for item in self.items if (kinds is None or item["kind"] in kinds)
                    and (roles is None or item["role"] in roles)
                    and (scope is None or item.get("scope") == str(scope))]
        return filtered[-n:]

    def retrieve_for_prompt(self, user_text, limit=5, scope=None):
        """Profile facts and explicit user facts are global; plain dialogue can be scoped (e.g. per session)."""
        if limit <= 0:
            return []
        profile = self.retrieve(user_text, n=max(limit // 2, 1), kinds=["profile_fact"])
        if scope is None:
            dialogue = self.retrieve(user_text, n=limit, kinds=["dialogue"], roles=["user"])
        else:
            facts = self.retrieve(user_text, n=max(limit // 2, 1), kinds=["user_fact"], roles=["user"])
            dialogue = facts + self.retrieve(user_text, n=limit, kinds=["dialogue"], roles=["user"], scope=scope)
        merged = []
        seen = set()
        for item in profile + dialogue:
            content = item.get("content")
            role = item.get("role")
            key = (role, item.get("kind"), content)
            if not content or content == user_text or key in seen:
                continue
            if item.get("kind") in ("dialogue", "user_fact") and role != "user":
                continue
            merged.append(item)
            seen.add(key)
            if len(merged) >= limit:
                break
        return merged
