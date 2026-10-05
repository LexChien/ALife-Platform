#!/usr/bin/env python3
"""Real-model verification harness for the 2026-10-05 gemma_web crash fix (commits 5c533d3 + f8bd8e0).

Talks to an already-running gemma_web (default http://127.0.0.1:8080); never starts/stops it.
Every subcommand writes <out>/<name>.json (machine-readable) and <out>/<name>.log (human-readable).

  stress      N concurrent /api/voice_chat (real WAVs from macOS `say` + afconvert) + concurrent /api/chat per round;
              PID before/after, new python*.ips crash reports in ~/Library/Logs/DiagnosticReports.
  embed       Separate process (NOT gemma_web): load PrefixedSentenceTransformerEF (same class/model/device as the
              config), hammer _encode from T threads x I iterations; time _ENCODE_LOCK acquisition (lock wait) and
              hold; compare each concurrent vector with a single-thread reference.
  profile     POST /api/profile through a sequence (e.g. yaying,digiclone); measure HTTP return time and the
              ack status transitions until ready; always ends on the profile that was active at start.
  transcribe  One long (<=90 s) varied zh-TW utterance -> /api/transcribe; wall time, transcript length, max
              repeated n-gram run.
  ngram       Unit-level collapse_repeated_ngrams check (any period 4..64) on synthetic loops.
  endpoints   HTTP codes for /, /api/health, /avatar.jpg, /generated_video-3.mp4, /mic_check; PID; tmux session.

Usage: .venv/bin/python tools/test_gemma_web_concurrency.py stress --out runs/tests/<dir> --rounds 2,3,4,4,3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Generic (non-persona, non-Yaying) zh-TW prompts spoken with the macOS system voice.
VOICE_PROMPTS = [
    "今天天氣怎麼樣？",
    "請用一句話介紹你自己。",
    "你記得我昨天說過什麼嗎？",
    "推薦一本適合週末看的書。",
    "幫我想一個晚餐的菜單。",
    "台北現在幾點了？",
    "說一個簡短的笑話給我聽。",
    "你覺得學程式最重要的是什麼？",
]
CHAT_PROMPTS = [
    "請根據你對我的記憶，摘要我最近在做的三件事。",
    "我們之前聊過哪些關於旅行的話題？",
    "幫我回想一下我喜歡吃什麼。",
    "你對我的工作習慣有什麼印象？",
    "整理一下我們上一次對話的重點。",
]
LONG_TEXT = (
    "早上七點，城市慢慢醒來，街角的早餐店已經排起長長的隊伍。"
    "公車司機打開廣播，播報今天的路況與天氣，提醒乘客記得帶傘。"
    "在辦公室裡，工程師們一邊喝咖啡，一邊討論新版本的測試計畫。"
    "中午時分，公園裡有人散步，有人在樹下看書，也有小朋友在追逐鴿子。"
    "下午的會議討論了預算、時程與人力安排，大家同意下週再檢討一次。"
    "傍晚下起小雨，騎腳踏車的人拉起雨衣，夜市的攤販忙著搭起遮雨棚。"
    "晚上回到家，打開窗戶，聽見遠處火車經過的聲音，心情也跟著放鬆下來。"
    "睡前整理明天的待辦事項：回覆郵件、繳交報告，還有記得打電話給家人。"
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def meta() -> dict:
    return {"machine": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "git_head": git_head(), "cwd": str(ROOT), "argv": sys.argv}


class Out:
    def __init__(self, out_dir: str, name: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log_path = self.dir / f"{name}.log"
        self._fh = self.log_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            print(line, flush=True)
            self._fh.write(line + "\n")
            self._fh.flush()

    def save(self, data: dict) -> Path:
        p = self.dir / f"{self.name}.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"wrote {p}")
        return p


def http(method: str, url: str, body: bytes | None = None, headers: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, data, round(time.time() - t0, 3), None
    except urllib.error.HTTPError as e:
        return e.code, e.read(), round(time.time() - t0, 3), f"HTTPError {e.code}"
    except Exception as e:
        return None, b"", round(time.time() - t0, 3), f"{type(e).__name__}: {e}"


def server_pid(port: int) -> int | None:
    try:
        out = subprocess.check_output(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], text=True).split()
        return int(out[0]) if out else None
    except Exception:
        return None


def crash_reports_since(t_start: float) -> list:
    d = os.path.expanduser("~/Library/Logs/DiagnosticReports")
    hits = []
    for p in glob.glob(os.path.join(d, "*.ips")) + glob.glob(os.path.join(d, "Retired", "*.ips")):
        b = os.path.basename(p).lower()
        if b.startswith("python") and os.path.getmtime(p) >= t_start:
            hits.append({"file": p, "mtime": time.strftime("%F %T %z", time.localtime(os.path.getmtime(p)))})
    return sorted(hits, key=lambda h: h["file"])


def make_wav(text: str, path: Path, voice: str = "Meijia", rate: int | None = None) -> float:
    aiff = path.with_suffix(".aiff")
    cmd = ["say", "-v", voice, "-o", str(aiff)] + (["-r", str(rate)] if rate else []) + [text]
    subprocess.run(cmd, check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True)
    aiff.unlink(missing_ok=True)
    info = subprocess.check_output(["afinfo", str(path)], text=True)
    m = re.search(r"estimated duration:\s*([\d.]+)", info)
    return float(m.group(1)) if m else -1.0


def stats(xs: list) -> dict | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    s = sorted(xs)
    p = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {"n": len(s), "min": round(s[0], 4), "max": round(s[-1], 4), "mean": round(statistics.mean(s), 4),
            "p50": round(p(0.5), 4), "p95": round(p(0.95), 4)}


# ----------------------------------------------------------------------------------------------- stress
def cmd_stress(a) -> int:
    o = Out(a.out, "stress")
    base = a.url.rstrip("/")
    t_start = time.time()
    o.log(f"stress start {now_iso()} meta={json.dumps(meta(), ensure_ascii=False)}")
    wav_dir = Path(a.out) / "stress_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for i, txt in enumerate(VOICE_PROMPTS):
        p = wav_dir / f"prompt_{i:02d}.wav"
        dur = make_wav(txt, p, voice=a.voice)
        wavs.append({"path": str(p), "text": txt, "duration_s": dur, "bytes": p.stat().st_size})
        o.log(f"wav {p.name} dur={dur:.2f}s text={txt}")
    pid0 = server_pid(a.port)
    o.log(f"PID before={pid0}")
    rounds = [int(x) for x in a.rounds.split(",") if x.strip()]
    results, round_info, k = [], [], 0

    def voice_req(idx: int, w: dict, rnd: int):
        st, body, wall, err = http("POST", f"{base}/api/voice_chat", Path(w["path"]).read_bytes(),
                                   {"Content-Type": "audio/wav", "X-Session-ID": f"stress-{int(t_start)}-r{rnd}-v{idx}"},
                                   timeout=a.timeout)
        r = {"kind": "voice_chat", "round": rnd, "idx": idx, "http": st, "client_wall_s": wall, "error": err,
             "prompt": w["text"]}
        try:
            j = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            j = {"_raw": body[:300].decode("utf-8", "replace")}
        tm = j.get("timings") or {}
        r.update(ok=j.get("ok"), transcript=j.get("transcript"), reply=(j.get("reply") or "")[:120],
                 server_error=j.get("error"), detail=j.get("detail"),
                 prep_retrieve_s=tm.get("prep_retrieve_s"), stt_s=tm.get("stt_s"), total_s=tm.get("total_s"),
                 timings=tm)
        return r

    def chat_req(idx: int, msg: str, rnd: int):
        payload = json.dumps({"session_id": f"stress-{int(t_start)}-r{rnd}-c{idx}", "message": msg}).encode()
        st, body, wall, err = http("POST", f"{base}/api/chat", payload, {"Content-Type": "application/json"},
                                   timeout=a.timeout)
        r = {"kind": "chat", "round": rnd, "idx": idx, "http": st, "client_wall_s": wall, "error": err, "prompt": msg}
        try:
            j = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            j = {}
        tm = j.get("timings") or {}
        r.update(ok=j.get("ok"), reply=(j.get("reply") or "")[:120], server_error=j.get("error"),
                 detail=j.get("detail"), prep_retrieve_s=tm.get("prep_retrieve_s"), total_s=tm.get("total_s"),
                 timings=tm)
        return r

    for rnd, conc in enumerate(rounds, 1):
        jobs = []
        t_r = time.time()
        o.log(f"round {rnd}: {conc} voice_chat + {a.chat_per_round} chat concurrently; pid={server_pid(a.port)}")
        with ThreadPoolExecutor(max_workers=conc + a.chat_per_round) as ex:
            for i in range(conc):
                jobs.append(ex.submit(voice_req, i, wavs[k % len(wavs)], rnd))
                k += 1
            for c in range(a.chat_per_round):
                jobs.append(ex.submit(chat_req, c, CHAT_PROMPTS[(rnd + c) % len(CHAT_PROMPTS)], rnd))
            rr = [j.result() for j in jobs]
        wall = round(time.time() - t_r, 3)
        pid_r = server_pid(a.port)
        for r in rr:
            o.log(f"  r{rnd} {r['kind']}#{r['idx']} http={r['http']} ok={r['ok']} wall={r['client_wall_s']}s "
                  f"prep_retrieve={r.get('prep_retrieve_s')} stt={r.get('stt_s')} total={r.get('total_s')} "
                  f"err={r['error'] or r.get('server_error')} transcript={r.get('transcript')!r}")
        o.log(f"round {rnd} wall={wall}s pid_after={pid_r}")
        round_info.append({"round": rnd, "voice_concurrency": conc, "chat": a.chat_per_round, "wall_s": wall,
                           "pid_after": pid_r})
        results.extend(rr)
        if pid_r != pid0:
            o.log("PID CHANGED -> stopping")
            break
        time.sleep(a.gap)
    pid1 = server_pid(a.port)
    crashes = crash_reports_since(t_start)
    voice = [r for r in results if r["kind"] == "voice_chat"]
    chat = [r for r in results if r["kind"] == "chat"]
    ok = lambda rs: sum(1 for r in rs if r["http"] == 200 and r["ok"] is True)
    summary = {
        "voice_requests": len(voice), "voice_ok": ok(voice), "chat_requests": len(chat), "chat_ok": ok(chat),
        "pid_before": pid0, "pid_after": pid1, "pid_unchanged": pid0 is not None and pid0 == pid1,
        "new_python_crash_reports": crashes,
        "voice_prep_retrieve_s": stats([r["prep_retrieve_s"] for r in voice]),
        "voice_stt_s": stats([r["stt_s"] for r in voice]), "voice_total_s": stats([r["total_s"] for r in voice]),
        "voice_client_wall_s": stats([r["client_wall_s"] for r in voice]),
        "chat_prep_retrieve_s": stats([r["prep_retrieve_s"] for r in chat]),
        "chat_total_s": stats([r["total_s"] for r in chat]),
        "round_wall_s": [ri["wall_s"] for ri in round_info],
    }
    summary["pass"] = (summary["voice_ok"] == summary["voice_requests"] > 0 and summary["chat_ok"] == summary["chat_requests"]
                       and summary["pid_unchanged"] and not crashes)
    o.log(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    o.save({"test": "stress", "start": time.strftime("%F %T %z", time.localtime(t_start)), "end": now_iso(),
            "meta": meta(), "rounds": round_info, "wavs": wavs, "results": results, "summary": summary})
    return 0 if summary["pass"] else 1


# ----------------------------------------------------------------------------------------------- embed
class TimedRLock:
    """Drop-in for embeddings._ENCODE_LOCK that records acquire wait and hold time per outermost acquisition."""

    def __init__(self):
        self._l = threading.RLock()
        self._tls = threading.local()
        self.waits: list = []
        self.holds: list = []
        self._rec = threading.Lock()

    def acquire(self, *args, **kw):
        t0 = time.perf_counter()
        got = self._l.acquire(*args, **kw)
        if got:
            depth = getattr(self._tls, "depth", 0)
            if depth == 0:
                self._tls.t_acq = time.perf_counter()
                with self._rec:
                    self.waits.append(self._tls.t_acq - t0)
            self._tls.depth = depth + 1
        return got

    def release(self):
        self._tls.depth -= 1
        if self._tls.depth == 0:
            with self._rec:
                self.holds.append(time.perf_counter() - self._tls.t_acq)
        self._l.release()

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()


def cmd_embed(a) -> int:
    o = Out(a.out, "embed")
    sys.path.insert(0, str(ROOT))
    o.log(f"embed start {now_iso()} meta={json.dumps(meta(), ensure_ascii=False)}")
    try:
        vm = subprocess.check_output(["memory_pressure"], text=True).strip().splitlines()[-1]
        sw = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        o.log(f"memory before: {vm} | {sw}")
    except Exception as exc:
        vm = sw = f"unavailable: {exc}"
    import numpy as np
    from digital_clone.memory import embeddings as E
    t0 = time.time()
    ef = E.PrefixedSentenceTransformerEF(a.model, device=a.device)
    load_s = round(time.time() - t0, 3)
    o.log(f"loaded {a.model} on {a.device} in {load_s}s; torch device={getattr(ef._model, 'device', '?')}")
    texts = [VOICE_PROMPTS[i % len(VOICE_PROMPTS)] + (" 補充說明" * (i % 7)) + CHAT_PROMPTS[i % len(CHAT_PROMPTS)]
             for i in range(16)]
    # Warm-up + single-thread reference (no contention) -> serial per-call baseline.
    ef.embed_query([texts[0]])
    ref, serial = {}, []
    for i, t in enumerate(texts):
        s = time.perf_counter()
        ref[("q", i)] = np.asarray(ef.embed_query([t])[0])
        serial.append(time.perf_counter() - s)
        s = time.perf_counter()
        ref[("d", i)] = np.asarray(ef([t])[0])
        serial.append(time.perf_counter() - s)
    o.log(f"serial baseline per call: {stats(serial)}")
    timed = TimedRLock()
    orig = E._ENCODE_LOCK
    E._ENCODE_LOCK = timed  # _encode looks the module global up at call time
    errors, mism, calls = [], [], []
    rec = threading.Lock()

    def worker(tid: int):
        for it in range(a.iters):
            i = (tid * 7 + it) % len(texts)
            kind = "q" if (tid + it) % 2 == 0 else "d"
            s = time.perf_counter()
            try:
                v = np.asarray((ef.embed_query if kind == "q" else ef)([texts[i]])[0])
                cos = float(np.dot(v, ref[(kind, i)]) / (np.linalg.norm(v) * np.linalg.norm(ref[(kind, i)])))
                with rec:
                    calls.append(time.perf_counter() - s)
                    if not np.isfinite(v).all() or cos < 0.999:
                        mism.append({"tid": tid, "it": it, "kind": kind, "i": i, "cos": cos})
            except Exception as exc:
                with rec:
                    errors.append(f"tid={tid} it={it}: {type(exc).__name__}: {exc}")

    t_c = time.time()
    ths = [threading.Thread(target=worker, args=(t,)) for t in range(a.threads)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = round(time.time() - t_c, 3)
    E._ENCODE_LOCK = orig
    total = a.threads * a.iters
    summary = {"model": a.model, "device": a.device, "load_s": load_s, "threads": a.threads, "iters": a.iters,
               "calls_expected": total, "calls_done": len(calls), "errors": len(errors), "mismatches_cos_lt_0.999": len(mism),
               "concurrent_wall_s": wall, "throughput_calls_per_s": round(len(calls) / wall, 2) if wall else None,
               "serial_call_s": stats(serial), "concurrent_call_latency_s": stats(calls),
               "lock_wait_s": stats(timed.waits), "lock_hold_s": stats(timed.holds),
               "lock_acquisitions": len(timed.waits), "memory_before": vm, "swap_before": sw}
    summary["pass"] = len(calls) == total and not errors and not mism and len(timed.waits) == total
    for e in errors[:20]:
        o.log(f"ERROR {e}")
    for m in mism[:20]:
        o.log(f"MISMATCH {m}")
    o.log(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    o.save({"test": "embed", "end": now_iso(), "meta": meta(), "summary": summary, "errors": errors,
            "mismatches": mism})
    return 0 if summary["pass"] else 1


# ----------------------------------------------------------------------------------------------- profile
def cmd_profile(a) -> int:
    o = Out(a.out, "profile")
    base = a.url.rstrip("/")
    o.log(f"profile start {now_iso()} meta={json.dumps(meta(), ensure_ascii=False)}")
    st, body, _, err = http("GET", f"{base}/api/profile", timeout=30)
    cur = json.loads(body)
    start_profile = cur.get("active")
    o.log(f"active at start={start_profile} ack={cur.get('ack')} tts_voice={cur.get('tts_voice')}")
    pid0 = server_pid(a.port)
    seq = [p for p in a.sequence.split(",") if p]
    if not seq or seq[-1] != start_profile:
        seq.append(start_profile)
    steps = []
    for p in seq:
        t0 = time.time()
        st, body, ret_s, err = http("POST", f"{base}/api/profile", json.dumps({"profile": p}).encode(),
                                    {"Content-Type": "application/json"}, timeout=a.post_timeout)
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        ack0 = (j.get("ack") or {}).get("status")
        o.log(f"POST profile={p} http={st} return_s={ret_s} active={j.get('active')} tts_voice={j.get('tts_voice')} ack={ack0} err={err}")
        trans = [{"t_s": ret_s, "status": ack0}]
        ready_s = 0.0 if ack0 == "ready" else None
        while ready_s is None and time.time() - t0 < a.ready_timeout:
            time.sleep(a.poll)
            _, b2, _, e2 = http("GET", f"{base}/api/profile", timeout=30)
            try:
                s2 = (json.loads(b2).get("ack") or {}).get("status")
            except Exception:
                s2 = f"poll_error {e2}"
            el = round(time.time() - t0, 2)
            if s2 != trans[-1]["status"]:
                trans.append({"t_s": el, "status": s2})
                o.log(f"  ack -> {s2} at {el}s")
            if s2 == "ready":
                ready_s = el
            if isinstance(s2, str) and s2.startswith("error"):
                break
        o.log(f"  profile={p} ready_s={ready_s} transitions={trans} pid={server_pid(a.port)}")
        steps.append({"profile": p, "http": st, "return_s": ret_s, "active": j.get("active"),
                      "tts_voice": j.get("tts_voice"), "ack_initial": ack0, "ack_ready_s": ready_s, "transitions": trans,
                      "error": err})
    _, body, _, _ = http("GET", f"{base}/api/profile", timeout=30)
    end = json.loads(body)
    pid1 = server_pid(a.port)
    summary = {"start_profile": start_profile, "end_profile": end.get("active"), "end_ack": (end.get("ack") or {}).get("status"),
               "pid_before": pid0, "pid_after": pid1, "steps": [{k: s[k] for k in ("profile", "http", "return_s", "ack_initial", "ack_ready_s")} for s in steps]}
    summary["pass"] = (all(s["http"] == 200 and s["active"] == s["profile"] and s["ack_ready_s"] is not None for s in steps)
                       and summary["end_profile"] == start_profile and pid0 == pid1 and summary["end_ack"] == "ready")
    o.log(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    o.save({"test": "profile", "end": now_iso(), "meta": meta(), "steps": steps, "summary": summary})
    return 0 if summary["pass"] else 1


# ----------------------------------------------------------------------------------------------- transcribe
def max_repeat_run(text: str, n: int = 8) -> int:
    best = 1
    for i in range(0, max(0, len(text) - n)):
        unit, r, j = text[i:i + n], 1, i + n
        while text[j:j + n] == unit:
            r += 1
            j += n
        best = max(best, r)
    return best


def cmd_transcribe(a) -> int:
    o = Out(a.out, "transcribe")
    base = a.url.rstrip("/")
    o.log(f"transcribe start {now_iso()} meta={json.dumps(meta(), ensure_ascii=False)}")
    wav = Path(a.out) / "transcribe_long.wav"
    text = LONG_TEXT * a.repeat_text
    dur = make_wav(text, wav, voice=a.voice, rate=a.rate)
    o.log(f"wav {wav} duration={dur:.2f}s bytes={wav.stat().st_size} source_chars={len(text)} voice={a.voice}")
    if dur > 90.0:
        o.log("audio > 90 s -> abort (limit)")
        return 2
    pid0 = server_pid(a.port)
    t_start = time.time()
    st, body, wall, err = http("POST", f"{base}/api/transcribe", wav.read_bytes(),
                               {"Content-Type": "audio/wav", "X-Session-ID": f"longtx-{int(t_start)}"}, timeout=a.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    tx = j.get("transcript") or ""
    pid1 = server_pid(a.port)
    summary = {"audio_s": dur, "source_chars": len(text), "http": st, "ok": j.get("ok"), "wall_s": wall,
               "transcript_len": len(tx), "transcript_to_source_ratio": round(len(tx) / len(text), 3),
               "max_repeat_run_8gram": max_repeat_run(tx), "pid_before": pid0, "pid_after": pid1,
               "new_python_crash_reports": crash_reports_since(t_start), "error": err or j.get("error")}
    summary["pass"] = (st == 200 and j.get("ok") is True and 0 < len(tx) <= 2 * len(text)
                       and summary["max_repeat_run_8gram"] < 3 and pid0 == pid1 and not summary["new_python_crash_reports"])
    o.log(f"TRANSCRIPT {tx}")
    o.log(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    o.save({"test": "transcribe", "end": now_iso(), "meta": meta(), "source_text": text, "transcript": tx,
            "summary": summary})
    return 0 if summary["pass"] else 1


# ----------------------------------------------------------------------------------------------- ngram
def cmd_ngram(a) -> int:
    o = Out(a.out, "ngram")
    sys.path.insert(0, str(ROOT))
    from genai.web.voice import collapse_repeated_ngrams
    head = "好的，我們今天來討論一下這個計畫的進度。"
    loop = "謝謝大家收看，請記得訂閱我的頻道。"
    tail = "下次見。"
    cases = {
        "zh_loop_x40": head + loop * 40 + tail,
        "en_loop_x30": "Okay so the plan is ready. " + "Thank you for watching. " * 30 + "Bye.",
        "normal_zh": LONG_TEXT,
        "huge_50k": "字" * 50000,
    }
    rows = []
    for name, s in cases.items():
        t0 = time.perf_counter()
        out = collapse_repeated_ngrams(s)
        el = time.perf_counter() - t0
        rows.append({"case": name, "in_len": len(s), "out_len": len(out), "elapsed_ms": round(el * 1000, 3),
                     "in_count_loop": s.count(loop) if "zh" in name else s.count("Thank you for watching."),
                     "out_count_loop": out.count(loop) if "zh" in name else out.count("Thank you for watching."),
                     "out_preview": out[:160]})
        o.log(f"{name}: in_len={len(s)} out_len={len(out)} {el*1000:.2f}ms out={out[:160]!r}")
    r = {x["case"]: x for x in rows}
    checks = {
        "zh_loop_collapsed": r["zh_loop_x40"]["out_len"] < 0.2 * r["zh_loop_x40"]["in_len"] and head in collapse_repeated_ngrams(cases["zh_loop_x40"]),
        "en_loop_collapsed": r["en_loop_x30"]["out_len"] < 0.3 * r["en_loop_x30"]["in_len"],
        "normal_unchanged": collapse_repeated_ngrams(LONG_TEXT) == LONG_TEXT.strip(),
        "huge_capped": r["huge_50k"]["out_len"] <= 4001,
    }
    # Period sweep with production defaults (min_period=4..max_period=64): all periods should collapse.
    sweep = {}
    base_chars = ("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黃宇宙洪荒日月盈昃"
                  "春秋冬夏往來寒暑秋收冬藏閏餘成歲律呂調陽")
    for period in range(4, 65):
        unit = base_chars[:period]
        s = "開頭說明。" + unit * 30 + "結尾。"
        out = collapse_repeated_ngrams(s)
        sweep[period] = {"in_len": len(s), "out_len": len(out), "collapsed": len(out) < 0.5 * len(s),
                         "out_equals_one": out == "開頭說明。" + unit + "結尾。"}
    collapsed = [p for p, v in sweep.items() if v["collapsed"]]
    not_collapsed = [p for p, v in sweep.items() if not v["collapsed"]]
    o.log(f"period sweep (min_period=4..max_period=64; unit*30): collapsed periods={collapsed}; "
          f"not collapsed={not_collapsed}")
    checks["period_sweep_all_collapsed"] = len(not_collapsed) == 0 and len(collapsed) == 61
    ok = all(checks.values())
    o.log(f"SUMMARY checks={checks} pass={ok}")
    o.save({"test": "ngram", "end": now_iso(), "meta": meta(), "rows": rows, "checks": checks,
            "period_sweep_defaults": sweep, "collapsed_periods": collapsed, "pass": ok})
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------- endpoints
def cmd_endpoints(a) -> int:
    o = Out(a.out, "endpoints")
    base = a.url.rstrip("/")
    o.log(f"endpoints start {now_iso()} meta={json.dumps(meta(), ensure_ascii=False)}")
    codes = {}
    for p in ["/", "/api/health", "/avatar.jpg", "/generated_video-3.mp4", "/mic_check"]:
        st, body, wall, err = http("GET", base + p, timeout=60)
        codes[p] = st
        o.log(f"GET {p} -> {st} bytes={len(body)} {wall}s {err or ''}")
    tmux = subprocess.run(["tmux", "has-session", "-t", a.tmux], capture_output=True).returncode == 0
    pid = server_pid(a.port)
    o.log(f"tmux has-session {a.tmux}: {tmux}; listen pid={pid} expect={a.expect_pid}")
    ok = all(c == 200 for c in codes.values()) and tmux and (a.expect_pid is None or pid == a.expect_pid)
    o.save({"test": "endpoints", "end": now_iso(), "meta": meta(), "codes": codes, "tmux": tmux, "pid": pid,
            "expect_pid": a.expect_pid, "pass": ok})
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--port", type=int, default=8080)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress"); s.add_argument("--out", required=True); s.add_argument("--rounds", default="2,3,4,4,3")
    s.add_argument("--chat-per-round", type=int, default=1); s.add_argument("--voice", default="Meijia")
    s.add_argument("--timeout", type=float, default=600); s.add_argument("--gap", type=float, default=2.0)
    e = sub.add_parser("embed"); e.add_argument("--out", required=True); e.add_argument("--model", default="BAAI/bge-m3")
    e.add_argument("--device", default="mps"); e.add_argument("--threads", type=int, default=8); e.add_argument("--iters", type=int, default=25)
    p = sub.add_parser("profile"); p.add_argument("--out", required=True); p.add_argument("--sequence", default="yaying,digiclone")
    p.add_argument("--ready-timeout", type=float, default=900); p.add_argument("--post-timeout", type=float, default=300)
    p.add_argument("--poll", type=float, default=1.0)
    t = sub.add_parser("transcribe"); t.add_argument("--out", required=True); t.add_argument("--voice", default="Meijia")
    t.add_argument("--rate", type=int, default=None); t.add_argument("--repeat-text", type=int, default=1)
    t.add_argument("--timeout", type=float, default=900)
    n = sub.add_parser("ngram"); n.add_argument("--out", required=True)
    d = sub.add_parser("endpoints"); d.add_argument("--out", required=True); d.add_argument("--tmux", default="gemma_web")
    d.add_argument("--expect-pid", type=int, default=None)
    a = ap.parse_args()
    return {"stress": cmd_stress, "embed": cmd_embed, "profile": cmd_profile, "transcribe": cmd_transcribe,
            "ngram": cmd_ngram, "endpoints": cmd_endpoints}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
