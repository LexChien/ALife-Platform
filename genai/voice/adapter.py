class DummyVoiceAdapter:
    def synthesize(self, text):
        return {"status": "stub", "text": text}
