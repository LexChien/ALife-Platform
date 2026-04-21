import re


class MemoryStore:
    def __init__(self):
        self.items = []

    def add(self, role, content, kind="dialogue"):
        self.items.append({"role": role, "content": content, "kind": kind})

    def recent(self, n=5):
        return self.items[-n:]

    def add_profile_facts(self, facts):
        for fact in facts:
            self.add("system", fact, kind="profile_fact")

    def retrieve(self, query, n=3, kinds=None):
        query_terms = self._terms(query)
        scored = []
        for idx, item in enumerate(self.items):
            if kinds is not None and item["kind"] not in kinds:
                continue
            item_terms = self._terms(item["content"])
            overlap = len(query_terms & item_terms)
            recency_bonus = (idx + 1) / max(len(self.items), 1) * 0.01
            kind_bonus = 0.05 if item["kind"] == "profile_fact" else 0.0
            score = overlap + recency_bonus + kind_bonus
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]

    def _terms(self, text):
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", text.lower())
        return {tok for tok in tokens if len(tok) >= 2}
