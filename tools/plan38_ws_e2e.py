#!/usr/bin/env python3
"""Plan 38 J2 E2E over the REAL realtime WebSocket (gemma_web :PORT+1/ws/session): synthetic `say -v Meijia`
utterances streamed as real-time 20 ms Int16 frames. Measures (server events, client clock):
  speech_end(audio) -> ack event, transcript, first `audio` event (first sentence synthesized)
and barge-in: a second utterance streamed while the server is `speaking` -> `stop` event latency from its onset.
Browser playback / echo cancellation are NOT included (Lex must test by hand). Source labelled SYNTHETIC say."""
import argparse, asyncio, json, statistics, subprocess, sys, tempfile, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def say(text, voice="Meijia"):
    d = Path(tempfile.mkdtemp()); a, w = d / "a.aiff", d / "a.wav"
    subprocess.run(["say", "-v", voice, "-o", str(a), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(a), str(w)], check=True)
    r = wave.open(str(w)); return np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)

UTTS = ["現在幾點了？", "幫我想一個晚餐的點子。", "什麼是人工生命？", "我今天有點累。", "跟我說一個簡短的笑話。",
        "東京是哪個國家的首都？", "給我一句鼓勵的話。", "你是誰？"]

async def run(port, n, barge):
    import websockets
    url = f"ws://127.0.0.1:{port}/ws/session"
    rows = []
    async with websockets.connect(url, max_size=2 ** 23) as ws:
        events = []
        async def reader():
            async for m in ws:
                ev = json.loads(m); ev["_t"] = time.perf_counter(); events.append(ev)
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "hello", "mode": "open", "thoughts": False}))
        silence = np.zeros(320, dtype=np.int16)
        async def stream(pcm, tail_s=0.8):
            frames = np.concatenate([pcm, np.zeros(int(16000 * tail_s), dtype=np.int16)])
            t_start = time.perf_counter()
            for i in range(0, len(frames), 320):
                await ws.send(frames[i:i + 320].tobytes())
                await asyncio.sleep(max(0, t_start + (i + 320) / 16000 - time.perf_counter()))
            return t_start, t_start + len(pcm) / 16000
        for k in range(n):
            text = UTTS[k % len(UTTS)]
            pcm = say(text)
            events.clear()
            t_start, t_end = await stream(pcm)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 25:
                if any(e["type"] == "trace" for e in events):
                    break
                await asyncio.sleep(0.02)
            first = lambda typ: next((e for e in events if e["type"] == typ), None)
            ack, tr, au, tc = first("ack"), first("transcript"), first("audio"), first("trace")
            row = {"text": text, "transcript": tr and tr.get("text"), "ack_s": ack and round(ack["_t"] - t_end, 3),
                   "transcript_s": tr and round(tr["_t"] - t_end, 3), "first_audio_s": au and round(au["_t"] - t_end, 3),
                   "server_marks": tc and tc.get("marks"), "stt": tc and tc.get("stt")}
            rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
            await ws.send(json.dumps({"type": "playback", "event": "ended", "turn_id": tc and tc.get("turn_id")}))
            for _ in range(25):
                await ws.send(silence.tobytes()); await asyncio.sleep(0.02)
        barges = []
        for k in range(barge):
            events.clear()
            await ws.send(json.dumps({"type": "text", "text": "請慢慢講一個關於星星的長故事。"}))
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 20 and not any(e["type"] == "audio" for e in events):
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.3)
            pcm = say("等一下，停。")
            t_on, _ = await stream(pcm, tail_s=0.6)
            stop = next((e for e in events if e["type"] == "stop"), None)
            b = {"stop": bool(stop), "stop_from_onset_ms": stop and round((stop["_t"] - t_on) * 1000),
                 "server_detect_ms": stop and stop.get("detect_ms")}
            barges.append(b); print("BARGE", json.dumps(b), flush=True)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 25 and not any(e["type"] == "trace" for e in events if e.get("_t", 0) > t_on + 1):
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "playback", "event": "stopped"}))
            for _ in range(50):
                await ws.send(silence.tobytes()); await asyncio.sleep(0.02)
        rt.cancel()
    q = lambda xs, p: (sorted(xs)[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None)
    fa = [r["first_audio_s"] for r in rows if r["first_audio_s"] is not None]
    rep = {"source": "REAL gemma_web WS (turbo STT, llama-server, resident Meijia TTS); input SYNTHETIC say -v Meijia",
           "n": len(rows), "first_audio_p50_s": q(fa, 0.5), "first_audio_p95_s": q(fa, 0.95),
           "ack_p50_s": q([r["ack_s"] for r in rows if r["ack_s"] is not None], 0.5),
           "transcript_p50_s": q([r["transcript_s"] for r in rows if r["transcript_s"] is not None], 0.5),
           "barge": barges, "barge_stop_p50_ms": q([b["stop_from_onset_ms"] for b in barges if b["stop"]], 0.5),
           "rows": rows, "note": "times from end of the utterance audio (client clock); endpoint silence included"}
    out = ROOT / "runs/plan38/ws_e2e" / time.strftime("%Y%m%d-%H%M%S") / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows",)}, ensure_ascii=False)); print("saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--barge", type=int, default=3)
    a = ap.parse_args(); asyncio.run(run(a.port, a.n, a.barge))
