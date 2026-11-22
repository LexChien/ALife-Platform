#!/usr/bin/env python3
"""Plan 44 M4-M7 + Plan 45 orchestrator for the Jetson (CPU + GPU concurrently).

ZH: GPU 依序跑 fullload_v2 → dt 驗證 → M4 → P8 → P7 → M6；同時 CPU 6 個行程跑 M5 譜系樹，
跑完後持續追加 M5 重複（filler）直到 GPU 完成，確保 CPU 不閒置。每個階段完成會寫 stage_<name>.done，
重跑時自動跳過已完成階段（供 tools/cell_sim_supervisor.sh 失敗重啟用）。PROGRESS.md 每次更新。
EN: GPU stages run in sequence while a 6-process CPU pool runs M5 lineage trees, then keeps adding
M5 replicate trees until the GPU finishes. Each stage writes stage_<name>.done and is skipped on
restart (used by tools/cell_sim_supervisor.sh). PROGRESS.md is rewritten on every update.

Author / 作者: Grok Bot ALife Project - Andreeva, 2026-11-23 (Asia/Taipei).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GPU_STAGES = ["min_validation", "fullload_v2", "dt_validation", "m4", "p8", "p7", "m6"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.state = {"stages": {}, "errors": [], "cpu": "not started"}
        f = run / "progress_state.json"
        if f.exists():
            self.state = json.loads(f.read_text())

    def stage(self, name: str, frac: float, note: str) -> None:
        name = name.lower()   # experiments report "M4"/"M6"/"P7"; the table keys are lower-case (fixed 2026-11-23)
        self.state["stages"][name] = {"frac": round(frac, 4), "note": note, "updated": now()}
        self.state["current"] = name
        self.write()

    def error(self, msg: str) -> None:
        self.state["errors"].append(f"{now()} {msg}")
        self.write()

    def write(self) -> None:
        (self.run / "progress_state.json").write_text(json.dumps(self.state, indent=1))
        done = sum(1 for s in GPU_STAGES if (self.run / f"stage_{s}.done").exists())
        cur = self.state.get("current", "-")
        cs = self.state["stages"].get(cur, {})
        lines = [
            "# PROGRESS — Plan 44 M4–M7 + Plan 45（Jetson, CPU+GPU）",
            "",
            "作者 / Author: Grok Bot ALife Project - Andreeva；本檔由 tools/run_cell_m4_m7.py 自動改寫 / rewritten automatically.",
            "",
            f"- 最後更新 / Last update: {now()} Asia/Taipei",
            f"- 目前階段 / Current stage: {cur} — {cs.get('note', '')}",
            f"- 本階段完成度 / Stage progress: {100 * cs.get('frac', 0):.1f}%",
            f"- GPU 階段完成 / GPU stages done: {done}/{len(GPU_STAGES)} ({', '.join(s for s in GPU_STAGES if (self.run / f'stage_{s}.done').exists()) or 'none'})",
            f"- CPU（M5 譜系 + filler）/ CPU (M5 lineage + filler): {self.state.get('cpu')}",
            f"- 錯誤 / Errors: {len(self.state['errors'])}" + ("".join(f"\n  - {e}" for e in self.state["errors"][-5:]) if self.state["errors"] else " none"),
            "",
            "| 階段 / Stage | 完成度 / Progress | 說明 / Note | 更新 / Updated |",
            "|---|---|---|---|",
        ]
        for s in GPU_STAGES + ["m5_cpu", "report"]:
            v = self.state["stages"].get(s)
            if v:
                lines.append(f"| {s} | {100 * v['frac']:.1f}% | {v['note']} | {v['updated']} |")
            else:
                lines.append(f"| {s} | 0% | 尚未開始 / not started | - |")
        (self.run / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def cpu_main(run: str, founders: int, depth: int, mu: float, sigma: float, stop_flag: str, q) -> None:
    """CPU side: M5 trees on all cores, then filler trees until the stop flag exists."""
    from cell.heredity import _tree_worker
    out = Path(run) / "m5_lineage.jsonl"
    filler = Path(run) / "m5_filler_lineage.jsonl"
    # one core is left to the GPU driver process (it launches graphs and syncs), so CPU workers +
    # GPU feeder = all cores busy without starving the GPU feeder
    procs = max((os.cpu_count() or 6) - 1, 1)
    t0 = time.perf_counter()
    done_main = (Path(run) / "stage_m5_cpu.done").exists()
    with mp.get_context("spawn").Pool(procs) as pool:
        if not done_main:
            n = 0
            with out.open("w") as fh:
                for recs in pool.imap_unordered(_tree_worker, [(i, depth, mu, sigma, 5000 + i) for i in range(founders)]):
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                    n += 1
                    q.put(("m5_cpu", n / founders, f"{n}/{founders} founder trees (depth {depth})"))
            (Path(run) / "stage_m5_cpu.done").write_text(json.dumps({"wall_s": time.perf_counter() - t0, "at": now()}))
        k = founders
        with filler.open("a") as fh:
            while not Path(stop_flag).exists():
                batch = [(k + i, depth, mu, sigma, 5000 + k + i) for i in range(procs)]
                for recs in pool.imap_unordered(_tree_worker, batch):
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                k += procs
                q.put(("cpu_filler", 0.0, f"filler trees so far: {k - founders}"))
    q.put(("cpu_done", 1.0, f"cpu finished at {now()}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--m4-batch", type=int, default=1024)
    ap.add_argument("--m4-generations", type=int, default=12)
    ap.add_argument("--m6-npop", type=int, default=1024)
    ap.add_argument("--m6-generations", type=int, default=2000)
    ap.add_argument("--m5-founders", type=int, default=192)
    ap.add_argument("--m5-depth", type=int, default=5)
    ap.add_argument("--fullload-batch", type=int, default=32768)
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    prog = Progress(run)
    stamp = run / "stages.jsonl"

    from cell import experiments as ex
    from cell.params import load_params
    p = load_params()
    stop_flag = run / "gpu_finished.flag"
    if stop_flag.exists():
        stop_flag.unlink()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    cpu = ctx.Process(target=cpu_main, args=(str(run), a.m5_founders, a.m5_depth, p["m5_mutation_rate_per_locus"],
                                             p["mutation_sigma"], str(stop_flag), q))
    cpu.start()

    def drain():
        while not q.empty():
            name, frac, note = q.get()
            if name in ("m5_cpu",):
                prog.state["cpu"] = note
                prog.stage("m5_cpu", frac, note)
            else:
                prog.state["cpu"] = note
                prog.write()

    def cb(stage, frac, note):
        drain()
        prog.stage(stage, frac, note)

    jobs = {
        "min_validation": lambda: ex.min_validation(run),
        "fullload_v2": lambda: ex.fullload_v2(run, batch=a.fullload_batch, progress=cb),
        "dt_validation": lambda: ex.dt_validation(run),
        "m4": lambda: ex.m4_lineages(run, batch=a.m4_batch, generations=a.m4_generations, progress=cb),
        "p8": lambda: ex.p8_death(run, progress=cb),
        "p7": lambda: ex.p7_response(run, progress=cb),
        "m6": lambda: ex.m6_evolution(run, n_pop=a.m6_npop, generations=a.m6_generations,
                                      mu=p["mutation_rate_per_locus"], sigma=p["mutation_sigma"],
                                      dt=p["evolution_dt_s"], progress=cb),
    }
    rc = 0
    for name in GPU_STAGES:
        marker = run / f"stage_{name}.done"
        if marker.exists():
            continue
        prog.stage(name, 0.0, "started")
        t0 = time.time()
        with stamp.open("a") as fh:
            fh.write(json.dumps({"stage": name, "event": "start", "at": now()}) + "\n")
        try:
            res = jobs[name]()
        except Exception as exc:
            prog.error(f"{name}: {exc!r}"[:400])
            (run / f"stage_{name}.error.txt").write_text(traceback.format_exc())
            rc = 2
            break
        if name == "m6":
            res["generations"] = a.m6_generations
            (run / "m6_meta.json").write_text(json.dumps(res, indent=1, default=float))
        with stamp.open("a") as fh:
            fh.write(json.dumps({"stage": name, "event": "end", "at": now(), "wall_s": time.time() - t0}) + "\n")
        marker.write_text(json.dumps({"at": now(), "wall_s": time.time() - t0, "result": res}, default=float)[:20000])
        prog.stage(name, 1.0, f"done in {time.time() - t0:.0f} s")
    stop_flag.write_text(now())
    while cpu.is_alive():
        drain(); time.sleep(5)
    drain()
    if rc == 0:
        from cell.report import compute_gates
        g = compute_gates(run)
        (run / "gates.json").write_text(json.dumps(g, indent=1, sort_keys=True, default=float))
        prog.stage("report", 1.0, "gates.json written; run `python -m cell.report <run> --verify` for M7")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
