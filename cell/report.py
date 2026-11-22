"""Plan 44 M7: recompute every gate from the raw JSONL of a run and write gates + SUMMARY.

ZH: 本模組只讀 run 目錄裡的原始 JSONL，不讀任何模擬中的記憶體值；M7 門檻＝在另一個
行程重新計算一次，結果必須與已寫出的 gates.json 完全相同。
EN: Reads only the raw JSONL files of a run directory. Gate M7 = a fresh process recomputes
every gate and must reproduce gates.json exactly.

Author / 作者: Grok Bot ALife Project - Andreeva, 2026-11-23 (Asia/Taipei).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _r(x, n=6):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), n)


def gate_m4(run: Path) -> Dict[str, Any]:
    divs = _read(run / "m4_divisions.jsonl")
    gens = _read(run / "m4_generations.jsonl")
    val = _read(run / "m4_min_validation.jsonl")
    if not divs:
        return {"status": "NOT RUN"}
    f = np.array([d["volume_left_fraction"] for d in divs])
    dev = np.abs(2.0 * f - 1.0)               # |V_left - V/2| / (V/2)
    centre = np.abs(f - 0.5)                   # site offset as a fraction of length
    ok_gens = 0
    for g in gens:
        if g["not_ready"] == 0 and not g["nan_state"] and g["dead"] == 0:
            ok_gens += 1
        else:
            break
    periods = np.array([d["min_period_s"] for d in divs if d["min_period_s"] is not None and math.isfinite(d["min_period_s"])])
    dbl = np.array([d["doubling_s"] for d in divs if d["was_ready"] and math.isfinite(d["doubling_s"])])
    out = {
        "n_divisions": len(divs), "n_lineages": len({d["slot"] for d in divs}),
        "consecutive_generations_without_collapse": ok_gens,
        "volume_deviation_p50": _r(np.percentile(dev, 50)), "volume_deviation_p95": _r(np.percentile(dev, 95)),
        "volume_deviation_max": _r(dev.max()), "fraction_divisions_dev_le_10pct": _r((dev <= 0.10).mean()),
        "site_offset_p95_of_length": _r(np.percentile(centre, 95)), "site_offset_max_of_length": _r(centre.max()),
        "fraction_sites_within_centre_10pct": _r((centre <= 0.10).mean()),
        "raw_argmin_readout_volume_deviation_p95": _r(np.percentile(np.abs(2.0 * np.array([d["site_argmin"] for d in divs]) - 1.0), 95)) if "site_argmin" in divs[0] else None,
        "readout_fallbacks": int(sum(1 for d in divs if d.get("readout_fallback"))),
        "min_period_median_s": _r(np.median(periods)) if periods.size else None,
        "min_period_p10_p90_s": [_r(np.percentile(periods, 10)), _r(np.percentile(periods, 90))] if periods.size else None,
        "mind_conservation_min_max": [_r(min(d["mind_conservation"] for d in divs)), _r(max(d["mind_conservation"] for d in divs))],
        "doubling_median_s": _r(np.median(dbl)) if dbl.size else None,
        "deterministic_validation": val,
    }
    out["gate_generations"] = "PASS" if ok_gens >= 10 else "FAIL"
    out["gate_volume_deviation_p95_le_10pct"] = "PASS" if out["volume_deviation_p95"] <= 0.10 else "FAIL"
    out["gate_site_within_centre_10pct_p95"] = "PASS" if out["site_offset_p95_of_length"] <= 0.10 else "FAIL"
    out["status"] = "PASS" if all(out[k] == "PASS" for k in ("gate_generations", "gate_volume_deviation_p95_le_10pct",
                                                               "gate_site_within_centre_10pct_p95")) else "FAIL"
    # Plan 45 P2 / P3 from the same records
    out["p2_self_made_lipid_fraction_final"] = _r(gens[-1]["self_made_lipid_fraction_mean"]) if gens else None
    sav = np.array([d["sa_v_at_division_um_inv"] for d in divs])
    out["p2_sa_v_at_division_min_max"] = [_r(sav.min()), _r(sav.max())]
    out["max_closure"] = _r(max(g["max_closure"] for g in gens)) if gens else None
    return out


def gate_m5(run: Path) -> Dict[str, Any]:
    recs = _read(run / "m5_lineage.jsonl")
    if not recs:
        return {"status": "NOT RUN"}
    from cell.heredity import verify_lineage
    from cell.params import load_params
    v = verify_lineage(recs)
    mu = load_params()["m5_mutation_rate_per_locus"]
    v.pop("errors", None)
    v["mutation_rate_set"] = mu
    v["mutation_rate_rel_error"] = _r(abs(v["mutation_rate_measured"] - mu) / mu) if v["mutation_rate_measured"] else None
    v["mutation_rate_measured"] = _r(v["mutation_rate_measured"])
    v["max_partition_conservation_error"] = _r(v["max_partition_conservation_error"], 12)
    n_founders = len({r["cell_uid"] for r in recs if r["parent_uid"] is None})
    v["n_founders"] = n_founders
    v["gate_genome_id_valid"] = "PASS" if v["n_genome_id_valid"] == v["n_records"] else "FAIL"
    v["gate_relations_valid"] = "PASS" if v["n_relations_valid"] == v["n_relations"] and v["n_relations"] > 0 else "FAIL"
    v["gate_depth_ge_3"] = "PASS" if v["max_depth_generations"] >= 3 else "FAIL"
    v["gate_partition_conserved"] = "PASS" if v["max_partition_conservation_error"] <= 1e-6 else "FAIL"
    v["gate_mutation_rate_within_10pct"] = "PASS" if (v["mutation_rate_rel_error"] is not None and v["mutation_rate_rel_error"] <= 0.10) else "FAIL"
    v["status"] = "PASS" if all(v[k] == "PASS" for k in v if k.startswith("gate_")) else "FAIL"
    return v


def _phase_doubling(rows, phase, from_gen):
    """Per-population division-weighted mean doubling over windows of a phase after `from_gen`."""
    sel = [r for r in rows if r["phase"] == phase]
    if not sel:
        return None
    P = len(sel[0]["gens"])
    num = np.zeros(P); den = np.zeros(P)
    for r in sel:
        g = np.array(r["gens"]); n = np.array(r["window_divisions"]); d = np.array(r["window_mean_doubling_s"])
        m = g >= from_gen
        num += np.where(m, n * d, 0.0); den += np.where(m, n, 0.0)
    return num / np.maximum(den, 1)


def gate_m6(run: Path) -> Dict[str, Any]:
    rows = _read(run / "m6_evolution.jsonl")
    meta = json.loads((run / "m6_meta.json").read_text()) if (run / "m6_meta.json").exists() else None
    if not rows or not meta:
        return {"status": "NOT RUN"}
    cond = np.array(meta["conditions"])
    anc = _phase_doubling(rows, "ancestral_mu0", 5)
    evo_last = _phase_doubling(rows, "evolution", meta["generations"] - 100)
    garden = _phase_doubling(rows, "evolved_mu0", 5)
    ref = anc[cond == "control"].mean()
    w_anc, w_end, w_gar = ref / anc, ref / evo_last, ref / garden
    wa_w, wg_w = w_anc[cond == "weakened"], w_gar[cond == "weakened"]
    wa_c, wg_c = w_anc[cond == "control"], w_gar[cond == "control"]
    sd_ctrl_anc = float(np.std(wa_c, ddof=1))
    drop = 1.0 - float(wa_w.mean())
    pooled = math.sqrt((np.var(wa_w, ddof=1) + np.var(wg_w, ddof=1)) / 2.0)
    rise_w = float(wg_w.mean() - wa_w.mean())
    pooled_c = math.sqrt((np.var(wa_c, ddof=1) + np.var(wg_c, ddof=1)) / 2.0)
    rise_c = float(wg_c.mean() - wa_c.mean())
    last = [r for r in rows if r["phase"] == "evolution"][-1]
    out = {
        "populations": len(cond), "replicates_per_condition": int((cond == "control").sum()),
        "generations_evolved_min": _r(min(last["gens"]), 2),
        "ancestral_doubling_s": [_r(x, 2) for x in anc], "evolved_garden_doubling_s": [_r(x, 2) for x in garden],
        "w_ancestral": [_r(x) for x in w_anc], "w_evolution_last100": [_r(x) for x in w_end],
        "w_evolved_garden": [_r(x) for x in w_gar],
        "weakened_fitness_drop": _r(drop), "weakened_drop_in_control_sd": _r(drop / max(sd_ctrl_anc, 1e-12), 2),
        "weakened_recovery_fraction_of_drop": _r(rise_w / drop) if drop > 0 else None,
        "weakened_rise_in_pooled_sd": _r(rise_w / max(pooled, 1e-12), 2),
        "control_rise_in_pooled_sd": _r(rise_c / max(pooled_c, 1e-12), 2),
        "theta_mean_final_by_population": last["theta_mean"],
        "mutation_rate_measured": _r(last["mutation_events"] / max(last["mutation_trials"], 1)),
        "mutation_rate_set": meta["mu"], "max_closure": _r(max(r["max_closure"] for r in rows)),
        "dead_cells_max": max(r["dead"] for r in rows),
    }
    out["gate_drop_measurable"] = "PASS" if drop > 0 and out["weakened_drop_in_control_sd"] >= 2 else "FAIL"
    out["gate_partial_recovery_le_2000_gens"] = "PASS" if (out["generations_evolved_min"] <= 2000.5 + 100
                                                           and rise_w > 0 and out["weakened_rise_in_pooled_sd"] >= 2) else "FAIL"
    out["status"] = "PASS" if out["gate_drop_measurable"] == "PASS" and out["gate_partial_recovery_le_2000_gens"] == "PASS" else "FAIL"
    out["p6_L_A5_rise_ge_2sd_5_replicates"] = "PASS" if (out["replicates_per_condition"] >= 5 and
                                                          out["weakened_rise_in_pooled_sd"] >= 2) else "FAIL"
    out["mutation_rate_rel_error"] = _r(abs(out["mutation_rate_measured"] - meta["mu"]) / meta["mu"])
    return out


def gate_dt(run: Path) -> Dict[str, Any]:
    rows = {str(r["dt_s"]): r for r in _read(run / "m6_dt_validation.jsonl")}
    if "1.0" not in rows or "10.0" not in rows:
        return {"status": "NOT RUN"}
    rel = abs(rows["10.0"]["median_s"] - rows["1.0"]["median_s"]) / rows["1.0"]["median_s"]
    return {"median_dt1_s": rows["1.0"]["median_s"], "median_dt10_s": rows["10.0"]["median_s"],
            "rel_diff": _r(rel), "status": "PASS" if rel <= 0.02 else "FAIL", "tolerance": 0.02}


def gate_p8(run: Path) -> Dict[str, Any]:
    rows = _read(run / "p8_death.jsonl")
    if not rows:
        return {"status": "NOT RUN"}
    by: Dict[str, Dict[str, int]] = {}
    for r in rows:
        k = f"starve={r['starve_s']}s,frac={r['death_frac']},steps={r['death_steps']}"
        b = by.setdefault(k, {"n": 0, "dead": 0, "revived": 0, "divided_after": 0})
        b["n"] += 1; b["dead"] += int(r["dead"]); b["revived"] += int(r["revived"])
        b["divided_after"] += int(r["divisions_after_starvation"] > 0)
    default = [r for r in rows if r["death_frac"] == 0.05 and r["death_steps"] == 60]
    dead_default = [r for r in default if r["dead"]]
    controls = [r for r in rows if r["starve_s"] == 0]
    out = {"cells": len(rows), "by_condition": by,
           "default_criterion_dead_trials": len(dead_default),
           "default_criterion_revivals": sum(int(r["revived"]) for r in dead_default),
           "all_criteria_revivals": sum(int(r["revived"]) for r in rows),
           "unstarved_control_deaths": sum(int(r["dead"]) for r in controls),
           "dead_cells_that_divided_later": sum(1 for r in rows if r["dead"] and r["divisions_after_starvation"] > 0)}
    out["status"] = "PASS" if (out["default_criterion_dead_trials"] >= 20 and out["default_criterion_revivals"] == 0
                               and out["unstarved_control_deaths"] == 0) else "FAIL"
    return out


def gate_p7(run: Path) -> Dict[str, Any]:
    rows = _read(run / "p7_response.jsonl")
    if not rows:
        return {"status": "NOT RUN"}
    pre = [r for r in rows if r["glucose"] >= 0.999]
    post = [r for r in rows if r["glucose"] < 0.999]
    t_step = post[0]["t_s"] - 10.0 if post else None
    a0 = st.mean(r["atp_mean"] for r in pre[-30:])
    a1 = st.mean(r["atp_mean"] for r in post[-60:])
    target = a0 + 0.632 * (a1 - a0)
    lat = None
    for r in post:
        if (a1 < a0 and r["atp_mean"] <= target) or (a1 >= a0 and r["atp_mean"] >= target):
            lat = r["t_s"] - t_step; break
    d_pre = [d for r in pre for d in r["doubling_s"]]
    d_post = [d for r in post if r["t_s"] > t_step + 2 * 6300 for d in r["doubling_s"]]
    return {"atp_before": _r(a0, 1), "atp_after": _r(a1, 1), "atp_response_latency_63pct_s": _r(lat, 1),
            "doubling_before_median_s": _r(np.median(d_pre), 1) if d_pre else None,
            "doubling_after_median_s": _r(np.median(d_post), 1) if d_post else None,
            "status": "MEASURED (Plan 45 sets no numeric gate for P7)"}


def gate_fullload(run: Path) -> Dict[str, Any]:
    rows = _read(run / "fullload_v2.jsonl")
    return rows[0] if rows else {"status": "NOT RUN"}


def provenance() -> Dict[str, Any]:
    from cell.params import load_params
    p = load_params()
    rep = p.provenance_report()
    rep["status"] = "PASS" if rep["fraction_estimate_unmeasured"] <= 0.40 else "FAIL (qualitative only, Plan 44 section 8)"
    rep["table"] = {k: {"value": v["value"], "unit": v.get("unit"), "source": v["source"]} for k, v in p.constants.items()}
    return rep


def compute_gates(run: Path) -> Dict[str, Any]:
    g = {"M4": gate_m4(run), "M5": gate_m5(run), "M6": gate_m6(run), "M6_dt_check": gate_dt(run),
         "P8": gate_p8(run), "P7": gate_p7(run), "fullload_v2": gate_fullload(run), "provenance": provenance()}
    m4, m5, m6, p8 = g["M4"], g["M5"], g["M6"], g["P8"]

    def st_(cond):
        return "PASS" if cond else "FAIL"
    life = {
        "P1_L-A1_atp_closure_le_1pct": st_(all((x.get("max_closure") is not None and x["max_closure"] <= 0.01) for x in (m4, m6))) if m4.get("max_closure") is not None and m6.get("max_closure") is not None else "NOT MEASURED",
        "P2_L-A2_self_made_membrane_ge_50pct": st_(m4.get("p2_self_made_lipid_fraction_final", 0) >= 0.5) if m4.get("p2_self_made_lipid_fraction_final") is not None else "NOT MEASURED",
        "P3_L-A3_ge_3_generations_and_dev_le_10pct": st_(m4.get("consecutive_generations_without_collapse", 0) >= 3 and m4.get("volume_deviation_p95", 1) <= 0.10) if "volume_deviation_p95" in m4 else "NOT MEASURED",
        "P4_L-A4_trace_and_mutation_rate": st_(m5.get("status") == "PASS") if m5.get("status") != "NOT RUN" else "NOT MEASURED",
        "P5_development_multicellularity": "NOT IN SCOPE (Lex decision E1: no multicellular development)",
        "P6_L-A5_fitness_rise_ge_2sd": m6.get("p6_L_A5_rise_ge_2sd_5_replicates", "NOT MEASURED"),
        "P7_response": "MEASURED (no numeric gate in Plan 45)" if g["P7"].get("atp_before") is not None else "NOT MEASURED",
        "P8_L-A6_death_irreversible": p8.get("status", "NOT MEASURED"),
        "E3_assembly_index": "NOT MEASURED (the model has no molecular-structure objects to compute an assembly index on)",
    }
    g["Plan45"] = life
    g["Plan45"]["alive_by_project_convention"] = (
        "YES (P1-P4, P8 and P6 all PASS)" if all(life[k] == "PASS" for k in (
            "P1_L-A1_atp_closure_le_1pct", "P2_L-A2_self_made_membrane_ge_50pct", "P3_L-A3_ge_3_generations_and_dev_le_10pct",
            "P4_L-A4_trace_and_mutation_rate", "P6_L-A5_fitness_rise_ge_2sd", "P8_L-A6_death_irreversible")) else "NO (see FAIL/NOT MEASURED items)")
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--write", action="store_true", help="write gates.json")
    ap.add_argument("--verify", action="store_true", help="recompute and compare with gates.json (gate M7)")
    a = ap.parse_args()
    run = Path(a.run)
    g = compute_gates(run)
    blob = json.dumps(g, indent=1, sort_keys=True, default=float)
    if a.write:
        (run / "gates.json").write_text(blob)
    if a.verify:
        stored = (run / "gates.json").read_text()
        same = stored == blob
        res = {"gate": "M7 recompute-from-raw-JSONL", "identical": same, "bytes": len(blob),
               "status": "PASS" if same else "FAIL"}
        (run / "gates_m7_verify.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return 0 if same else 1
    print(blob[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
