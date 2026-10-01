#!/usr/bin/env python3
"""Plan 37 V3/E5/C1: live HTTP eval against the running gemma_web (real llama.cpp Gemma).

Voice inputs are SYNTHETIC speech generated with macOS `say` (labelled as such);
they exercise the real STT -> LLM -> TTS path but are not human recordings.

R2 (2026-10-02): empathy is gated by a rubric judge run with the real local Gemma
(genai.web.emotion_llm.judge_empathy_llm; A=emotion recognised, B=feelings addressed,
C=helpful; each 0-2, pass = total>=4 and A>=1). The round-1 keyword marker is still
recorded as `empathy_marker` but no longer gates. Also: harder emotional prompts
(E5b, label reported, empathy gated), Lenia factual check, chunked-TTS checks and
time-to-first-audio (stt_s + llm_s + first_chunk_s).
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
from core.config import load_config  # noqa: E402
from genai.llm.factory import create_llm_adapter  # noqa: E402
from genai.web.emotion_llm import judge_empathy_llm  # noqa: E402

_JUDGE = {}


def judge(user_text, reply):
    if "a" not in _JUDGE:
        _JUDGE["a"] = create_llm_adapter(load_config(str(ROOT / "configs/genai/gemma_llama_cpp.yaml"), profile="mac_metal"))
    return judge_empathy_llm(_JUDGE["a"], user_text, reply)


def empathy_pass(j):
    return bool(j and j.get("scores") and j["total"] >= 4 and j["scores"]["A"] >= 1)

BASE = "http://127.0.0.1:8080"
EMPATHY = ["抱歉", "理解", "聽起來", "辛苦", "難過", "陪", "感受", "心情", "不容易", "擔心", "沒關係", "可以理解", "生氣", "焦慮", "緊張", "替你", "恭喜", "開心", "太棒", "高興", "了不起", "很棒", "真棒"]
EMO_CASES = [
    ("sadness", "我今天很難過，工作被主管當眾罵了，覺得自己很沒用。"),
    ("anger", "氣死我了！客戶又臨時改需求，我整個週末都白做了！"),
    ("fear", "我好擔心明天的面試，緊張到睡不著。"),
    ("joy", "太好了！我的 Lenia 實驗終於成功跑出穩定的生命體了！"),
    ("neutral", "請用一句話說明 Lenia 是什麼。"),
]
EMO_HARD = [  # E5b: harder phrasing; label reported, empathy (rubric) gated
    ("anger", "太好了，又要加班到半夜，真是開心死了。"),
    ("sadness", "我養了十二年的狗昨天走了。"),
    ("fear", "明天要開刀，醫生說成功率只有一半。"),
    ("sadness", "心好累，什麼都不想做。"),
    ("joy", "爽啦！終於抽到演唱會門票！"),
    ("fear", "我很高興他回來了，可是我好怕他又會離開。"),
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
        rubric = judge(text, reply) if expected != "neutral" else None
        row = {"expected": expected, "input": text, "http": status, "elapsed_s": elapsed, "reply": reply,
               "observed": p["emotion"]["observed"]["label"], "state": p["emotion"]["state"]["label"],
               "guidance_applied": bool(p["emotion"]["modulation"]["system_guidance"]),
               "tts": {k: tts.get(k) for k in ("ok", "wpm", "duration_s", "voice")}, "wav_http": wav_status,
               "hygiene": hyg, "server_hygiene": p.get("hygiene"), "empathy_marker": empathy,
               "empathy_rubric": rubric, "dna": p.get("dna"),
               "tts_chunks": len(tts.get("chunks") or []), "tts_first_chunk_s": tts.get("first_chunk_s")}
        if expected == "neutral":
            row["factual_lenia_ok"] = ("細胞自動機" in reply or "cellular automat" in reply.lower()) and "對話層" not in reply
        row["pass"] = (status == 200 and bool(reply.strip()) and row["observed"] == expected and not any(hyg.values())
                       and tts.get("ok") and wav_status == 200 and (expected == "neutral" or empathy_pass(rubric))
                       and row.get("factual_lenia_ok", True))
        emo_rows.append(row)
        print("E5", expected, "PASS" if row["pass"] else "FAIL", elapsed, "s wpm", tts.get("wpm"), "|", reply.replace("\n", " ")[:160])
    report["emotion_cases"] = emo_rows

    hard_rows = []
    for expected, text in EMO_HARD:
        sid = f"e5b-{uuid.uuid4().hex[:6]}"
        status, p = post_json("/api/chat", {"session_id": sid, "message": text})
        reply = p.get("reply", "")
        rubric = judge(text, reply)
        row = {"expected": expected, "input": text, "http": status, "reply": reply,
               "observed": p["emotion"]["observed"]["label"], "label_ok": p["emotion"]["observed"]["label"] == expected,
               "empathy_rubric": rubric, "hygiene": hygiene(reply)}
        row["pass"] = status == 200 and bool(reply.strip()) and empathy_pass(rubric) and not any(row["hygiene"].values())
        hard_rows.append(row)
        print("E5b", expected, row["observed"], "PASS" if row["pass"] else "FAIL", rubric and rubric.get("scores"), "|", reply.replace("\n", " ")[:120])
    report["emotion_hard_cases"] = hard_rows

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
        chunk_status = [get(c["url"])[0] for c in (tts.get("chunks") or [])]
        tm = p.get("timings") or {}
        ttfa = (round(tm.get("stt_s", 0) + tm.get("llm_s", 0) + (tts.get("first_chunk_s") or 0), 3)
                if tm.get("llm_s") is not None else None)
        row = {"input_audio": "SYNTHETIC macOS say voice=" + voice, "reference": text, "transcript": p.get("transcript"),
               "cer": cer(text, p.get("transcript", "")), "stt_provider": p.get("stt_provider"), "reply": reply,
               "emotion_observed": (p.get("emotion") or {}).get("observed", {}).get("label"),
               "voice_features": (p.get("emotion") or {}).get("voice_features"), "voice_arousal": (p.get("emotion") or {}).get("voice_arousal"),
               "tts_ok": tts.get("ok"), "tts_wav_http": wav_status, "timings": p.get("timings"), "elapsed_s": elapsed,
               "tts_chunk_http": chunk_status, "time_to_first_audio_s": ttfa,
               "http": status, "hygiene": hygiene(reply)}
        row["pass"] = status == 200 and row["cer"] <= 0.15 and bool(reply.strip()) and not any(row["hygiene"].values()) and wav_status == 200 and all(c == 200 for c in chunk_status)
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
        "leaks_total": sum(any(r["hygiene"].values()) for r in emo_rows + voice_rows + hard_rows) + sum(any(t["reply_hygiene"].values()) for t in traj),
        "mean_voice_cer": round(sum(r["cer"] for r in voice_rows) / len(voice_rows), 3),
        "emotion_hard_pass": sum(r["pass"] for r in hard_rows), "emotion_hard_n": len(hard_rows),
        "emotion_hard_label_ok": sum(r["label_ok"] for r in hard_rows),
        "mean_total_s": round(sum((r["timings"] or {}).get("total_s", 0) for r in voice_rows) / len(voice_rows), 3),
        "mean_time_to_first_audio_s": round(sum(r["time_to_first_audio_s"] or 0 for r in voice_rows) / len(voice_rows), 3),
        "stt_provider": voice_rows[0]["stt_provider"] if voice_rows else None,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPORT", out / "report.json")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
