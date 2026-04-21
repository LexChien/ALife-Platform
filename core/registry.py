class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name):
        def deco(obj):
            self._items[name] = obj
            return obj
        return deco

    def create(self, name, **kwargs):
        if name not in self._items:
            raise ValueError(f"Unknown component: {name}. Available: {list(self._items.keys())}")
        return self._items[name](**kwargs)

    def list(self):
        return list(self._items.keys())
