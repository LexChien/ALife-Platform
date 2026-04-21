class DummyLLMAdapter:
    def generate(self, prompt, context=None):
        return f"[DummyLLM] prompt={prompt} context={context or 'none'}"
