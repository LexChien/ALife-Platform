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
ap.add_argument("--engine", required=True, choices=["f5", "f5mlx", "cosyvoice2", "gsv"])
ap.add_argument("--ref", required=True); ap.add_argument("--ref-text", required=True)
ap.add_argument("--device", default="mps"); ap.add_argument("--nfe", type=int, default=16)
ap.add_argument("--method", default="euler", help="f5mlx ODE solver (euler: NFE = steps)")
ap.add_argument("--gsv-root", default=os.path.expanduser("~/yaying_cache/GPT-SoVITS"))
ap.add_argument("--gsv-gpt", default=""); ap.add_argument("--gsv-sovits", default="")
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
elif A.engine == "f5mlx":
    # F5-TTS on Apple MLX (lucasnewman/f5-tts-mlx): model + 24 kHz reference loaded ONCE (cached reference)
    os.environ.setdefault("F5MLX_COMPILE", "0")  # measured 2026-10-03: per-call mx.compile adds ~1 s (shapes vary per text)
    import mlx.core as mx, librosa
    from f5_tts_mlx.cfm import F5TTS as F5MLX
    from f5_tts_mlx.utils import convert_char_to_pinyin
    from f5_tts_mlx.generate import estimated_duration, FRAMES_PER_SEC, TARGET_RMS
    eng = F5MLX.from_pretrained("lucasnewman/f5-tts-mlx")
    y, _ = librosa.load(A.ref, sr=24000, mono=True)
    REF = mx.array(y.astype(np.float32)); rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < TARGET_RMS: REF = REF * (TARGET_RMS / rms)
    REF_TEXT = A.ref_text.strip()

    def synth(text, speed):
        t = time.time()
        dur = int(estimated_duration(REF, REF_TEXT, text, speed) * FRAMES_PER_SEC)
        wave, _ = eng.sample(mx.expand_dims(REF, axis=0), text=convert_char_to_pinyin([REF_TEXT + " " + text]),
                             duration=dur, steps=A.nfe, method=A.method, speed=speed, cfg_strength=2.0,
                             sway_sampling_coef=-1.0, seed=None)
        wave = wave[REF.shape[0]:]; mx.eval(wave)
        el = time.time() - t
        return np.array(wave, dtype=np.float32), 24000, el, el
    synth("你好。", 1.0)
elif A.engine == "gsv":
    # GPT-SoVITS v2 fine-tuned on the clean 雅英 dataset (weights private, machine-local)
    import glob
    if not A.gsv_gpt:  # newest fine-tuned weights (private, machine-local)
        A.gsv_gpt = max(glob.glob(os.path.join(A.gsv_root, "GPT_weights_v2", "*.ckpt")), key=os.path.getmtime)
    if not A.gsv_sovits:
        A.gsv_sovits = max(glob.glob(os.path.join(A.gsv_root, "SoVITS_weights_v2", "*.pth")), key=os.path.getmtime)
    A.ref = os.path.abspath(A.ref)
    os.chdir(A.gsv_root); sys.path.insert(0, A.gsv_root); sys.path.insert(0, os.path.join(A.gsv_root, "GPT_SoVITS"))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    cfg = TTS_Config({"custom": {"device": A.device, "is_half": False, "version": "v2",
                                  "t2s_weights_path": A.gsv_gpt, "vits_weights_path": A.gsv_sovits,
                                  "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                  "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"}})
    eng = TTS(cfg)

    def synth(text, speed):
        t = time.time(); parts = []; first = None
        lang = "all_zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
        for sr, chunk in eng.run({"text": text, "text_lang": lang, "ref_audio_path": A.ref, "prompt_text": A.ref_text,
                                  "prompt_lang": "all_zh", "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                                  "text_split_method": "cut0", "batch_size": 1, "speed_factor": speed,
                                  "streaming_mode": False, "parallel_infer": False, "return_fragment": False}):
            if first is None: first = time.time() - t
            parts.append(np.asarray(chunk, np.float32) / (32768.0 if np.asarray(chunk).dtype == np.int16 else 1.0))
        return np.concatenate(parts), sr, time.time() - t, first
    synth("你好。", 1.0)
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
out_stream.write(json.dumps({"ready": True, "engine": A.engine, **({"gsv_gpt": os.path.basename(A.gsv_gpt), "gsv_sovits": os.path.basename(A.gsv_sovits)} if A.engine == "gsv" else {}), "load_s": round(time.time() - t0, 2), "device": A.device}) + "\n")
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
