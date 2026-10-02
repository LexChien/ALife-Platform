#!/usr/bin/env python3
"""Plan 40 T4: resident voice-clone TTS worker (runs inside the engine's own venv, NOT the gemma_web venv).

  <engine venv>/bin/python voice/clone_tts_worker.py --engine f5|cosyvoice2 --ref ref.wav --ref-text "..." [--device mps]

Protocol (JSON lines, same shape as voice/native/tts_daemon.swift):
  stdout first line: {"ready": true, "engine": ..., "load_s": ...}
  stdin : {"id": "...", "text": "...", "out": "/abs/out.wav", "speed": 1.0}
  stdout: {"id": "...", "ok": true, "duration_s": .., "synth_s": .., "first_audio_s": .., "sr": ..}
first_audio_s = time until the first audio samples exist (CosyVoice2 stream=True first chunk; F5 = whole utterance,
it is non-autoregressive). Model weights stay private/machine-local (never committed)."""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, choices=["f5", "cosyvoice2"])
ap.add_argument("--ref", required=True); ap.add_argument("--ref-text", required=True)
ap.add_argument("--device", default="mps"); ap.add_argument("--nfe", type=int, default=16)
ap.add_argument("--cosyvoice-root", default=os.path.expanduser("~/yaying_cache/CosyVoice"))
A = ap.parse_args()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, soundfile as sf  # noqa: E402

out_stream = sys.stdout
sys.stdout = sys.stderr  # engine prints must never corrupt the protocol
t0 = time.time()
if A.engine == "f5":
    from f5_tts.api import F5TTS
    eng = F5TTS(model="F5TTS_v1_Base", device=A.device)
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    REF, REF_TEXT = preprocess_ref_audio_text(A.ref, A.ref_text)

    def synth(text, speed):
        t = time.time()
        wav, sr, _ = eng.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=text, nfe_step=A.nfe, speed=speed,
                               show_info=lambda *a, **k: None, progress=None, remove_silence=False)
        el = time.time() - t
        return np.asarray(wav, np.float32), sr, el, el
    synth("你好。", 1.0)  # warm-up (MPS kernels)
else:
    sys.path.insert(0, A.cosyvoice_root); sys.path.insert(0, os.path.join(A.cosyvoice_root, "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice2
    eng = CosyVoice2(os.path.join(A.cosyvoice_root, "pretrained_models/CosyVoice2-0.5B"), load_jit=False, load_trt=False, fp16=False)
    assert eng.add_zero_shot_spk(A.ref_text, A.ref, "yaying") is True

    def synth(text, speed):
        t = time.time(); first = None; parts = []
        for j in eng.inference_zero_shot(text, "", "", zero_shot_spk_id="yaying", stream=True, speed=speed):
            if first is None: first = time.time() - t
            parts.append(j["tts_speech"].squeeze(0).cpu().numpy())
        return np.concatenate(parts).astype(np.float32), eng.sample_rate, time.time() - t, first
    synth("你好。", 1.0)
out_stream.write(json.dumps({"ready": True, "engine": A.engine, "load_s": round(time.time() - t0, 2), "device": A.device}) + "\n")
out_stream.flush()
for line in sys.stdin:
    try:
        req = json.loads(line)
    except Exception:
        continue
    try:
        wav, sr, el, first = synth(req["text"], float(req.get("speed", 1.0)))
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if peak > 0.99: wav = wav / peak * 0.95
        sf.write(req["out"], wav, sr, subtype="PCM_16")
        msg = {"id": req.get("id"), "ok": True, "duration_s": round(len(wav) / sr, 3), "synth_s": round(el, 3),
               "first_audio_s": round(first, 3), "sr": sr}
    except Exception as exc:  # report, keep serving
        msg = {"id": req.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out_stream.write(json.dumps(msg, ensure_ascii=False) + "\n"); out_stream.flush()
