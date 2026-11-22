#!/usr/bin/env python3
"""Summarise a tegrastats log: per-core CPU load/frequency, GPU load, RAM, temperature, power."""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP (\d+)/(\d+)MB")
TJ_RE = re.compile(r"tj@([\d.]+)C")
GPUT_RE = re.compile(r"gpu@([\d.]+)C")
VDD_RE = re.compile(r"VDD_IN (\d+)mW")


def parse(path: Path) -> dict:
    cores: list[list[float]] = []
    freqs: list[list[float]] = []
    gr3d, ram, swap, tj, gput, vdd = [], [], [], [], [], []
    n = 0
    for line in path.read_text(errors="ignore").splitlines():
        m = CPU_RE.search(line)
        if not m:
            continue
        n += 1
        loads, fs = [], []
        for part in m.group(1).split(","):
            part = part.strip()
            if part in ("off",):
                loads.append(0.0); fs.append(0.0); continue
            load, _, freq = part.partition("@")
            loads.append(float(load.rstrip("%")))
            fs.append(float(freq) if freq else 0.0)
        cores.append(loads); freqs.append(fs)
        g = GR3D_RE.search(line)
        if g: gr3d.append(float(g.group(1)))
        r = RAM_RE.search(line)
        if r: ram.append((float(r.group(1)), float(r.group(2))))
        s = SWAP_RE.search(line)
        if s: swap.append(float(s.group(1)))
        t = TJ_RE.search(line)
        if t: tj.append(float(t.group(1)))
        gt = GPUT_RE.search(line)
        if gt: gput.append(float(gt.group(1)))
        v = VDD_RE.search(line)
        if v: vdd.append(float(v.group(1)))
    if not cores:
        return {"samples": 0}
    ncore = len(cores[0])
    per_core_mean = [st.mean(row[i] for row in cores) for i in range(ncore)]
    per_core_max = [max(row[i] for row in cores) for i in range(ncore)]
    per_core_freq_mean = [st.mean(row[i] for row in freqs) for i in range(ncore)]
    per_core_freq_max = [max(row[i] for row in freqs) for i in range(ncore)]
    return {
        "samples": n,
        "cpu_cores": ncore,
        "cpu_load_mean_pct_per_core": [round(x, 1) for x in per_core_mean],
        "cpu_load_max_pct_per_core": per_core_max,
        "cpu_load_mean_pct_all": round(st.mean(per_core_mean), 1),
        "cpu_freq_mean_mhz_per_core": [round(x) for x in per_core_freq_mean],
        "cpu_freq_max_mhz_per_core": [round(x) for x in per_core_freq_max],
        "gpu_gr3d_mean_pct": round(st.mean(gr3d), 1) if gr3d else None,
        "gpu_gr3d_max_pct": max(gr3d) if gr3d else None,
        "ram_used_max_mb": max(x[0] for x in ram) if ram else None,
        "ram_total_mb": ram[0][1] if ram else None,
        "swap_used_max_mb": max(swap) if swap else None,
        "tj_max_c": max(tj) if tj else None,
        "gpu_temp_max_c": max(gput) if gput else None,
        "vdd_in_mean_mw": round(st.mean(vdd)) if vdd else None,
        "vdd_in_max_mw": max(vdd) if vdd else None,
    }


if __name__ == "__main__":
    out = parse(Path(sys.argv[1]))
    print(json.dumps(out, indent=2))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding="utf-8")
