import re
import numpy as np
from collections import defaultdict

class MemoryStore:
    def __init__(self):
        self.items = []
        self.vocab = {}
        self.idf = {}

    def _update_idf(self):
        self.vocab = {}
        doc_counts = defaultdict(int)
        for i, item in enumerate(self.items):
            tokens = self._tokenize(item["content"])
            for tok in set(tokens):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                doc_counts[tok] += 1
        
        N = max(len(self.items), 1)
        self.idf = {tok: np.log(N / (count + 1)) + 1.0 for tok, count in doc_counts.items()}

    def _vectorize(self, text):
        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocab))
        if not self.vocab: 
            return vec
        for tok in tokens:
            if tok in self.vocab:
                vec[self.vocab[tok]] += self.idf.get(tok, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(self, role, content, kind="dialogue"):
        self.items.append({"role": role, "content": content, "kind": kind})
        self._update_idf()

    def recent(self, n=5):
        return self.items[-n:]

    def add_profile_facts(self, facts):
        for fact in facts:
            self.add("system", fact, kind="profile_fact")

    def retrieve(self, query, n=3, kinds=None):
        if not self.items: 
            return []
            
        q_vec = self._vectorize(query)
        scored = []
        
        for idx, item in enumerate(self.items):
            if kinds is not None and item["kind"] not in kinds:
                continue
                
            i_vec = self._vectorize(item["content"])
            overlap = float(np.dot(q_vec, i_vec))
            
            recency_bonus = (idx + 1) / max(len(self.items), 1) * 0.05
            kind_bonus = 0.1 if item["kind"] == "profile_fact" else 0.0
            
            score = overlap + recency_bonus + kind_bonus
            # Lower threshold to allow fuzzy matches
            if score > 0.02: 
                scored.append((score, item))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]

    def _tokenize(self, text):
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", text.lower())
        return [tok for tok in tokens if len(tok) >= 2]
