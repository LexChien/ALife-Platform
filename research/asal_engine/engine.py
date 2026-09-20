from pathlib import Path

import numpy as np

from core.artifacts import save_gif, save_image, save_mp4
from core.logger import save_json
from .provenance import SCORE_ATOL, provenance_snapshot, sha256_file
from .scoring import TrajectoryScorer, resolve_config
from .search.optim import evo_search
from .substrates import substrates


class ASALEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)

    def run(self):
        cfg = resolve_config(self.config)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        scorer = TrajectoryScorer(cfg)
        narrative_cfg = cfg.get("narrative", {})
        morphology_cfg = cfg.get("morphology_judge", {})
        substrate = substrates.create(cfg["substrate"]["name"], **cfg["substrate"].get("params", {}))
        narrative_enabled = bool(narrative_cfg.get("enabled", False))
        if hasattr(substrate, "configure_narrative") and narrative_enabled:
            substrate.configure_narrative(cfg["runtime"]["steps"], narrative_cfg.get("phases", []))
        search = cfg["search"]
        seed = search["seed"]
        low = np.array(search["theta_low"], dtype=float)
        high = np.array(search["theta_high"], dtype=float)
        rs = np.random.RandomState(seed)
        init = [rs.uniform(low, high) for _ in range(search["keep"])]

        captured_states = {}
        trajectory_stats = []

        def simulate(theta, collect_frames=False, collect_states=False):
            substrate.reset(theta, seed=seed)
            frames = []
            for _ in range(cfg["runtime"]["steps"]):
                substrate.step(cfg["runtime"]["substeps"])
                if collect_frames:
                    frames.append(np.asarray(substrate.render()).copy())
                if collect_states:
                    for field in ("positions", "velocities", "grid"):
                        value = getattr(substrate, field, None)
                        if isinstance(value, np.ndarray):
                            captured_states.setdefault(field, []).append(value.copy())
                    if hasattr(substrate, "stats"):
                        trajectory_stats.append(substrate.stats())
            return frames if collect_frames else [np.asarray(substrate.render())]

        def eval_theta(theta):
            scores, _ = scorer.score(simulate(theta, collect_frames=narrative_enabled))
            return scores["combined"]

        _, _, best, best_score = evo_search(
            init, eval_theta, iters=search["iters"], pop=search["pop"],
            keep=search["keep"], sigma=search["sigma"], bounds=(low, high), seed=seed,
        )
        frames = simulate(best, collect_frames=True, collect_states=True)
        scores, narrative_result = scorer.score(frames)
        scores.update(search_best_score=float(best_score),
                      search_replay_delta=abs(float(best_score) - scores["combined"]),
                      search_replay_matches=bool(np.isclose(best_score, scores["combined"], atol=SCORE_ATOL, rtol=0)),
                      absolute_tolerance=SCORE_ATOL)
        np.savez_compressed(self.run_dir / "trajectory.npz", frames=np.stack(frames))
        state_name = "simulation_states.npz" if captured_states else None
        if captured_states:
            np.savez_compressed(self.run_dir / state_name,
                                **{key: np.stack(value) for key, value in captured_states.items()})
        save_json(self.run_dir / "substrate_stats.json", trajectory_stats)
        save_json(self.run_dir / "resolved_config.json", cfg)
        save_json(self.run_dir / "scores.json", scores)
        manifest = {
            "schema_version": "1.0", "seed": seed, "best_theta": np.asarray(best).tolist(),
            "trajectory": {"file": "trajectory.npz", "shape": list(np.stack(frames).shape),
                           "dtype": str(frames[0].dtype)},
            "config": "resolved_config.json", "scores": "scores.json",
            "score_formula": "research.asal_engine.scoring.TrajectoryScorer",
            "absolute_tolerance": SCORE_ATOL,
            "files_sha256": {name: sha256_file(self.run_dir / name) for name in
                             ("trajectory.npz", "resolved_config.json", "scores.json")},
            "provenance": provenance_snapshot(),
        }
        for name in (state_name, "substrate_stats.json"):
            if name:
                manifest["files_sha256"][name] = sha256_file(self.run_dir / name)
        save_json(self.run_dir / "replay_manifest.json", manifest)
        if not scores["search_replay_matches"]:
            raise RuntimeError("Winning trajectory score does not match search; inspect scores.json")

        save_image(self.run_dir / "best.png", frames[-1])
        save_gif(self.run_dir / "best.gif", frames, fps=8)
        mp4_name = "best.mp4"
        mp4_error = None
        try:
            save_mp4(self.run_dir / mp4_name, frames, fps=8)
        except Exception as exc:
            mp4_name = None
            mp4_error = f"{type(exc).__name__}: {exc}"
        for name in ("best.png", "best.gif", mp4_name):
            if name:
                manifest["files_sha256"][name] = sha256_file(self.run_dir / name)
        narrative_summary_name = trajectory_stats_name = None
        narrative_keyframes = {}
        if narrative_result is not None:
            keyframe_dir = self.run_dir / "phase_keyframes"
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            for idx, phase in enumerate(narrative_result["phases"], start=1):
                name = f"phase{idx}_{phase['name']}.png"
                save_image(keyframe_dir / name, frames[phase["frame_index"]])
                narrative_keyframes[phase["name"]] = str(Path("phase_keyframes") / name)
            narrative_summary_name = "narrative_summary.json"
            trajectory_stats_name = "trajectory_morphology.json"
            save_json(self.run_dir / narrative_summary_name,
                      {key: value for key, value in narrative_result.items() if key != "frame_stats"})
            save_json(self.run_dir / trajectory_stats_name, narrative_result["frame_stats"])
            for name in (narrative_summary_name, trajectory_stats_name):
                manifest["files_sha256"][name] = sha256_file(self.run_dir / name)
            save_json(self.run_dir / "replay_manifest.json", manifest)

        result = {
            "mode": "asal_target", "prompt": cfg["prompt"], "best_score": float(best_score),
            "best_theta": np.asarray(best).tolist(), "num_frames": len(frames), "seed": seed,
            "foundation_model": cfg["foundation_model"]["name"], "substrate": cfg["substrate"]["name"],
            "morphology_judge": morphology_cfg.get("name") if morphology_cfg.get("enabled", False) else None,
            "morphology_weight": scorer.morphology_weight,
            "search_iters": int(search["iters"]), "search_pop": int(search["pop"]), "search_keep": int(search["keep"]),
            "gif": "best.gif", "mp4": mp4_name, "mp4_error": mp4_error,
            "raw_trajectory": "trajectory.npz", "resolved_config": "resolved_config.json",
            "simulation_states": state_name, "substrate_stats": "substrate_stats.json",
            "substrate_final_stats": trajectory_stats[-1] if trajectory_stats else None,
            "scores": "scores.json", "replay_manifest": "replay_manifest.json",
            "replay_matches": scores["search_replay_matches"], "score_components": scores["components"],
            "narrative_enabled": narrative_enabled, "narrative_summary": narrative_summary_name,
            "trajectory_stats": trajectory_stats_name, "narrative_keyframes": narrative_keyframes,
            "narrative_score": float(narrative_result["total_score"]) if narrative_result else None,
            "narrative_phase_order_valid": bool(narrative_result["phase_order_valid"]) if narrative_result else None,
            "narrative_accepted": bool(narrative_result["accepted"]) if narrative_result else None,
            "narrative_failure_reasons": narrative_result["failure_reasons"] if narrative_result else [],
        }
        save_json(self.run_dir / "replay_manifest.json", manifest)
        save_json(self.run_dir / "engine_result.json", result)
        return result
