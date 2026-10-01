#!/usr/bin/env python3
"""Plan 37 V3/E5/C1: live HTTP eval against the running gemma_web (real llama.cpp Gemma).

Voice inputs are SYNTHETIC speech generated with macOS `say` (labelled as such);
they exercise the real STT -> LLM -> TTS path but are not human recordings.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genai.llm.reasoning import has_prompt_echo_residue, has_reasoning_leak, looks_like_cli_banner  # noqa: E402

BASE = "http://127.0.0.1:8080"
EMPATHY = ["抱歉", "理解", "聽起來", "辛苦", "難過", "陪", "感受", "心情", "不容易", "擔心", "沒關係", "可以理解", "生氣", "焦慮", "緊張", "替你", "恭喜", "開心", "太棒", "高興", "了不起", "很棒", "真棒"]
EMO_CASES = [
    ("sadness", "我今天很難過，工作被主管當眾罵了，覺得自己很沒用。"),
    ("anger", "氣死我了！客戶又臨時改需求，我整個週末都白做了！"),
    ("fear", "我好擔心明天的面試，緊張到睡不著。"),
    ("joy", "太好了！我的 Lenia 實驗終於成功跑出穩定的生命體了！"),
    ("neutral", "請用一句話說明 Lenia 是什麼。"),
]
VOICE_CASES = [
    ("Meijia", "我今天很難過，可以陪我聊聊嗎？"),
    ("Flo (中文（台灣）)", "請用一句話介紹你自己。"),
    ("Eddy (中文（台灣）)", "我好擔心明天的報告，怎麼辦？"),
]


def post_json(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def post_audio(path, data, session_id, ctype="audio/wav", timeout=300):
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": ctype, "X-Session-ID": session_id})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def clean_chars(s):
    return re.sub(r"[\s\W_]+", "", s or "")


def cer(ref, hyp):
    a, b = clean_chars(ref), clean_chars(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return round(d[len(b)] / max(1, len(a)), 3)


def hygiene(text):
    return {"leak": has_reasoning_leak(text), "banner": looks_like_cli_banner(text), "echo": has_prompt_echo_residue(text)}


def main():
    out = ROOT / "runs/plan37/web_eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report = {"label": "REAL_MODEL live gemma_web (llama.cpp Gemma GGUF)", "base": BASE}
    _, _, health = get("/api/health")
    h = json.loads(health)
    report["health"] = {k: h.get(k) for k in ("profile", "llm_healthcheck", "stt", "tts", "emotion", "dna", "vlm")}

    emo_rows = []
    for expected, text in EMO_CASES:
        sid = f"e5-{expected}-{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        status, p = post_json("/api/chat", {"session_id": sid, "message": text, "tts": True})
        elapsed = round(time.time() - t0, 2)
        reply = p.get("reply", "")
        tts = p.get("tts") or {}
        wav_status = None
        if tts.get("ok"):
            wav_status, ctype, body = get(tts["url"])
            (out / f"{expected}.wav").write_bytes(body)
        hyg = hygiene(reply)
        empathy = any(w in reply for w in EMPATHY) if expected != "neutral" else None
        row = {"expected": expected, "input": text, "http": status, "elapsed_s": elapsed, "reply": reply,
               "observed": p["emotion"]["observed"]["label"], "state": p["emotion"]["state"]["label"],
               "guidance_applied": bool(p["emotion"]["modulation"]["system_guidance"]),
               "tts": {k: tts.get(k) for k in ("ok", "wpm", "duration_s", "voice")}, "wav_http": wav_status,
               "hygiene": hyg, "server_hygiene": p.get("hygiene"), "empathy_marker": empathy,
               "dna": p.get("dna")}
        row["pass"] = (status == 200 and bool(reply.strip()) and row["observed"] == expected and not any(hyg.values())
                       and tts.get("ok") and wav_status == 200 and (empathy is None or empathy))
        emo_rows.append(row)
        print("E5", expected, "PASS" if row["pass"] else "FAIL", elapsed, "s wpm", tts.get("wpm"), "|", reply.replace("\n", " ")[:160])
    report["emotion_cases"] = emo_rows

    # multi-turn decay on one session
    sid = f"e3-decay-{uuid.uuid4().hex[:6]}"
    traj = []
    for text in ["我今天真的好難過，好想哭。", "今天天氣如何？", "幫我列出三種人工生命基質。", "謝謝。"]:
        _, p = post_json("/api/chat", {"session_id": sid, "message": text})
        st = p["emotion"]["state"]
        traj.append({"input": text, "label": st["label"], "valence": st["valence"], "intensity": st["intensity"], "reply_hygiene": hygiene(p["reply"])})
        print("E3", st["label"], st["valence"], st["intensity"], "|", p["reply"].replace("\n", " ")[:80])
    report["decay_trajectory"] = traj
    report["decay_ok"] = traj[0]["label"] == "sadness" and traj[-1]["intensity"] < traj[0]["intensity"]

    voice_rows = []
    for voice, text in VOICE_CASES:
        aiff = out / f"voice_{len(voice_rows)}.aiff"
        wav = out / f"voice_{len(voice_rows)}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        sid = f"v3-{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        status, p = post_audio("/api/voice_chat", wav.read_bytes(), sid)
        elapsed = round(time.time() - t0, 2)
        reply = p.get("reply", "")
        tts = p.get("tts") or {}
        wav_status = get(tts["url"])[0] if tts.get("ok") else None
        row = {"input_audio": "SYNTHETIC macOS say voice=" + voice, "reference": text, "transcript": p.get("transcript"),
               "cer": cer(text, p.get("transcript", "")), "stt_provider": p.get("stt_provider"), "reply": reply,
               "emotion_observed": (p.get("emotion") or {}).get("observed", {}).get("label"),
               "voice_features": (p.get("emotion") or {}).get("voice_features"), "voice_arousal": (p.get("emotion") or {}).get("voice_arousal"),
               "tts_ok": tts.get("ok"), "tts_wav_http": wav_status, "timings": p.get("timings"), "elapsed_s": elapsed,
               "http": status, "hygiene": hygiene(reply)}
        row["pass"] = status == 200 and row["cer"] <= 0.15 and bool(reply.strip()) and not any(row["hygiene"].values()) and wav_status == 200
        voice_rows.append(row)
        print("V3", voice, "PASS" if row["pass"] else "FAIL", "cer", row["cer"], "timings", p.get("timings"), "|", p.get("transcript"), "->", reply.replace("\n", " ")[:100])
    report["voice_cases"] = voice_rows

    _, _, life = get("/api/life")
    live = json.loads(life).get("live_state")
    report["live_state_after"] = live
    report["summary"] = {
        "emotion_pass": sum(r["pass"] for r in emo_rows), "emotion_n": len(emo_rows),
        "voice_pass": sum(r["pass"] for r in voice_rows), "voice_n": len(voice_rows),
        "decay_ok": report["decay_ok"],
        "leaks_total": sum(any(r["hygiene"].values()) for r in emo_rows + voice_rows) + sum(any(t["reply_hygiene"].values()) for t in traj),
        "mean_voice_cer": round(sum(r["cer"] for r in voice_rows) / len(voice_rows), 3),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPORT", out / "report.json")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
