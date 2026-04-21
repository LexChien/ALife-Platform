class MemoryStore:
    def __init__(self):
        self.items = []

    def add(self, role, content):
        self.items.append({"role": role, "content": content})

    def recent(self, n=5):
        return self.items[-n:]
