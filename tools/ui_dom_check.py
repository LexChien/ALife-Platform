#!/usr/bin/env python3
"""Plan 38 J0.3/J0.4/J3.4 headless DOM check of the live gemma_web UI (real Chrome via the DevTools protocol).

Checks: (A4) no hue-rotate/saturate/sepia/invert/grayscale on the face layer or any ancestor for every emotion;
(J0.4) the ASAL #lifeVisual is not inside the avatar viewport and no opaque media element sits above the face;
(J3.4) thought-stream toggle defaults OFF. Saves a screenshot of the avatar frame for the appearance guard.
usage: ui_dom_check.py [--url http://127.0.0.1:8080/] [--out runs/plan38/ui_check/<ts>]"""
import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
EMOTIONS = ["neutral", "concerned", "reassuring", "calm", "smile", "attentive"]
JS = r"""
(() => {
  const bad = /hue-rotate|saturate|sepia|invert|grayscale/;
  const face = document.getElementById('avatarFace');
  const stage = document.getElementById('avatarStage') || document.body;
  const viewport = face ? face.closest('.noir-viewport') : null;
  const out = {face: !!face, emotions: {}, life_visual_in_viewport: null, covering: [], thought_toggle: null};
  if (!face) return out;
  face.scrollIntoView({block: 'center'});
  for (const emo of EMOS) {
    stage.dataset.emotion = emo;
    const chain = [];
    for (let el = face; el && el !== document.documentElement; el = el.parentElement) {
      const f = getComputedStyle(el).filter;
      if (f && f !== 'none') chain.push({el: el.id || el.className, filter: f});
    }
    out.emotions[emo] = {filters: chain, forbidden: chain.filter(c => bad.test(c.filter))};
  }
  stage.dataset.emotion = 'neutral';
  const lv = document.getElementById('lifeVisual');
  out.life_visual_in_viewport = !!(lv && viewport && viewport.contains(lv));
  const r = face.getBoundingClientRect();
  out.face_rect = {x: r.x, y: r.y, w: r.width, h: r.height};
  const pts = [[0.5, 0.35], [0.5, 0.5], [0.35, 0.45], [0.65, 0.45]];
  for (const [fx, fy] of pts) {
    const stack = document.elementsFromPoint(r.x + r.width * fx, r.y + r.height * fy);
    const idx = stack.indexOf(face);
    out.face_hit_index = out.face_hit_index ?? idx;
    for (const el of stack.slice(0, idx < 0 ? stack.length : idx)) {
      const tag = el.tagName;
      const allowedOverlay = el.dataset && el.dataset.faceOverlay === 'identity-guarded';
      if (['IMG', 'VIDEO', 'CANVAS'].includes(tag) && !allowedOverlay && getComputedStyle(el).visibility !== 'hidden'
          && !el.hidden && Number(getComputedStyle(el).opacity) > 0.05)
        out.covering.push({tag, id: el.id, cls: el.className});
    }
    if (idx < 0) out.covering.push({tag: 'FACE_NOT_IN_STACK', at: [fx, fy]});
  }
  const tt = document.getElementById('thoughtToggle');
  out.thought_toggle = tt ? {exists: true, checked: !!tt.checked} : {exists: false};
  return out;
})()
"""


async def run(url: str, out: Path, mouth_level: float | None = None) -> dict:
    import websockets
    port = 9333
    prof = tempfile.mkdtemp(prefix="chrome_ui_check_")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             "--no-first-run", "--window-size=1400,1100", "--autoplay-policy=no-user-gesture-required",
                             "--use-fake-ui-for-media-stream", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(150):  # up to 30 s: Chrome start is slow when the Mac is loaded
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                page = next(t for t in targets if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            mid = 0

            async def cmd(method, **params):
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == mid:
                        return msg.get("result", msg)
            await cmd("Page.enable")
            await cmd("Page.navigate", url=url)
            await asyncio.sleep(4.0)
            res = await cmd("Runtime.evaluate", expression=JS.replace("EMOS", json.dumps(EMOTIONS)), returnByValue=True)
            result = res["result"]["value"]
            if mouth_level is not None:  # Plan 38 J3: render an open-mouth keyframe to guard the SPEAKING frame
                lv = await cmd("Runtime.evaluate", returnByValue=True, expression=(
                    f"(function(){{if(!window.DigiAvatar)return {{ok:false}};window.DigiAvatar.setLevel({mouth_level});"
                    "const m=document.getElementById('avatarMouth');return {ok:window.DigiAvatar.isReady(),k:m.dataset.k||null,"
                    "opacity:getComputedStyle(m).opacity,src:m.getAttribute('src')};})()"))
                result["mouth"] = lv["result"]["value"]
                await asyncio.sleep(0.5)
            r = result.get("face_rect")
            if r:
                shot = await cmd("Page.captureScreenshot", format="png",
                                 clip={"x": r["x"], "y": r["y"], "width": r["w"], "height": r["h"], "scale": 1})
                (out / "avatar_frame.png").write_bytes(base64.b64decode(shot["data"]))
            full = await cmd("Page.captureScreenshot", format="png")
            (out / "page.png").write_bytes(base64.b64decode(full["data"]))
            return result
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/")
    ap.add_argument("--out", default=str(ROOT / "runs/plan38/ui_check" / time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--mouth-level", type=float, default=None, help="open the lip-sync layer (0-1) before the screenshot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run(a.url, out, a.mouth_level))
    forbidden = {e: v["forbidden"] for e, v in result.get("emotions", {}).items() if v["forbidden"]}
    checks = {"face_present": result.get("face"), "no_forbidden_filter_on_face": not forbidden,
              "life_visual_not_in_viewport": result.get("life_visual_in_viewport") is False,
              "face_not_covered": not result.get("covering"),
              "thought_toggle_default_off": (result.get("thought_toggle") or {}).get("checked") is False
              if (result.get("thought_toggle") or {}).get("exists") else None}
    if a.mouth_level is not None:
        checks["mouth_keyframe_shown"] = bool((result.get("mouth") or {}).get("ok")) and (result.get("mouth") or {}).get("k") not in (None, "0")
    report = {"kind": "REAL_HEADLESS_CHROME_DOM_CHECK", "url": a.url, "checks": checks, "raw": result,
              "ok": all(v is not False for v in checks.values())}
    (out / "dom_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(checks), "->", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
