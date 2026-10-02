"""Plan 38 J2.3: utterance STT on Apple GPU (mlx-whisper) for VAD-endpointed segments, zh output -> zh-TW (OpenCC).

Model is configurable: ``mlx-community/whisper-large-v3-turbo`` (fast) or ``mlx-community/whisper-large-v3-mlx``
(Plan 37 R2 accuracy choice). J2 measures both on the same clips and the config records the decision."""
from __future__ import annotations

import time

import numpy as np

TURBO = "mlx-community/whisper-large-v3-turbo"
LARGE_V3 = "mlx-community/whisper-large-v3-mlx"


class UtteranceSTT:
    def __init__(self, model_repo: str = TURBO, language: str | None = None, to_traditional: bool = True):
        self.model_repo = model_repo
        self.language = language
        self.to_traditional = to_traditional
        self._cc = None
        self.loaded = False

    @staticmethod
    def available() -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _opencc(self):
        if self._cc is None and self.to_traditional:
            try:
                from opencc import OpenCC
                self._cc = OpenCC("s2twp")
            except Exception:
                self._cc = False
        return self._cc or None

    def preload(self) -> float:
        t0 = time.time()
        self.transcribe(np.zeros(16000, dtype=np.float32))
        self.loaded = True
        return round(time.time() - t0, 3)

    def _model(self):
        if getattr(self, "_m", None) is None:
            import mlx.core as mx
            from mlx_whisper.load_models import load_model
            self._m = load_model(self.model_repo, dtype=mx.float16)  # own instance (not the global ModelHolder)
        return self._m

    def transcribe(self, pcm16k: np.ndarray, initial_prompt: str | None = None, languages=("zh", "en")) -> dict:
        """Single encoder pass: encode once, detect language over ``languages`` only (Lex speaks zh/en), decode from
        the same features. mlx_whisper.transcribe() with language=None encodes twice (~2x slower, measured in J2)."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions, decode, detect_language
        x = np.asarray(pcm16k, dtype=np.float32)
        if np.abs(x).max(initial=0) > 1.5:  # int16 scale
            x = x / 32768.0
        t0 = time.time()
        model = self._model()
        mel = log_mel_spectrogram(x, n_mels=model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        feats = model.encoder(mel[None])
        lang = self.language
        probs = None
        if lang is None:
            _, probs = detect_language(model, feats)
            probs = probs[0]
            cands = {k: v for k, v in probs.items() if not languages or k in languages}
            lang = max(cands, key=cands.get)
        opts = DecodingOptions(language=lang, temperature=0.0, fp16=True, without_timestamps=True,
                               prompt=initial_prompt)
        res = decode(model, feats, opts)
        res = res[0] if isinstance(res, list) else res
        text = (res.text or "").strip()
        if res.no_speech_prob > 0.8 and res.avg_logprob < -1.0:
            text = ""
        cc = self._opencc()
        if cc and lang == "zh":
            text = cc.convert(text)
        return {"text": text, "language": lang, "stt_s": round(time.time() - t0, 3), "model": self.model_repo,
                "audio_s": round(len(x) / 16000, 3), "no_speech_prob": round(float(res.no_speech_prob), 3),
                "lang_prob": round(float(probs[lang]), 3) if probs else None}
