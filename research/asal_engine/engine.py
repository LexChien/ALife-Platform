import numpy as np
from pathlib import Path
from foundation_models import foundation_models
from core.artifacts import save_gif, save_image, save_mp4
from .scores import supervised_target_score
from .narrative_scores import score_narrative_trajectory
from .search.optim import evo_search
from .substrates import substrates

class ASALEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir

    def run(self):
        cfg = self.config
        fm = foundation_models.create(cfg["foundation_model"]["name"], **cfg["foundation_model"].get("params", {}))
        narrative_cfg = cfg.get("narrative", {})
        morphology_cfg = cfg.get("morphology_judge")
        morphology_fm = None
        morphology_txt_emb = None
        morphology_weight = 0.0
        if morphology_cfg and morphology_cfg.get("enabled", False):
            morphology_fm = foundation_models.create(
                morphology_cfg["name"],
                **morphology_cfg.get("params", {}),
            )
            morphology_txt_emb = morphology_fm.txt_embed(cfg["prompt"])
            morphology_weight = float(morphology_cfg.get("weight", 0.0))
        substrate = substrates.create(cfg["substrate"]["name"], **cfg["substrate"].get("params", {}))
        if hasattr(substrate, "configure_narrative") and narrative_cfg.get("enabled", False):
            substrate.configure_narrative(cfg["runtime"]["steps"], narrative_cfg.get("phases", []))
        txt_emb = fm.txt_embed(cfg["prompt"])

        search = cfg["search"]
        low = np.array(search["theta_low"], dtype=float)
        high = np.array(search["theta_high"], dtype=float)
        rs = np.random.RandomState(search.get("seed", 0))
        init = [rs.uniform(low, high) for _ in range(search["keep"])]

        def combine_scores(semantic_score, morphology_score=None, narrative_score=None):
            if narrative_cfg.get("enabled", False):
                semantic_weight = float(narrative_cfg.get("semantic_weight", 0.2))
                morph_weight = float(narrative_cfg.get("morphology_weight", 0.2 if morphology_score is not None else 0.0))
                traj_weight = float(narrative_cfg.get("weight", 0.6 if narrative_score is not None else 0.0))
                weighted = []
                if semantic_score is not None and semantic_weight > 0.0:
                    weighted.append((semantic_weight, semantic_score))
                if morphology_score is not None and morph_weight > 0.0:
                    weighted.append((morph_weight, morphology_score))
                if narrative_score is not None and traj_weight > 0.0:
                    weighted.append((traj_weight, narrative_score))
                total_weight = sum(weight for weight, _ in weighted)
                if total_weight > 0.0:
                    return float(sum(weight * value for weight, value in weighted) / total_weight)
            score = semantic_score
            if morphology_score is not None and morphology_weight > 0.0:
                score = (1.0 - morphology_weight) * score + morphology_weight * morphology_score
            return float(score)

        def simulate(theta, collect_frames=False):
            substrate.reset(theta)
            frames = []
            for _ in range(cfg["runtime"]["steps"]):
                substrate.step(cfg["runtime"]["substeps"])
                if collect_frames:
                    frames.append(np.asarray(substrate.render()).copy())
            if collect_frames:
                return frames
            return np.asarray(substrate.render())

        def eval_theta(theta):
            frames = simulate(theta, collect_frames=bool(narrative_cfg.get("enabled", False)))
            if isinstance(frames, list):
                img = frames[-1]
            else:
                img = frames
            score = supervised_target_score([fm.img_embed(img)], txt_emb)
            morph_score = None
            if morphology_fm is not None and morphology_weight > 0.0:
                morph_score = supervised_target_score([morphology_fm.img_embed(img)], morphology_txt_emb)
            narrative_score = None
            if narrative_cfg.get("enabled", False):
                narrative_score = score_narrative_trajectory(frames, narrative_cfg)["total_score"]
            return combine_scores(score, morph_score, narrative_score)

        pool, scores, best, best_score = evo_search(
            init, eval_theta,
            iters=search["iters"],
            pop=search["pop"],
            keep=search["keep"],
            sigma=search["sigma"],
            bounds=(low, high),
        )

        frames = simulate(best, collect_frames=True)
        best_img = frames[-1]
        save_image(self.run_dir / "best.png", best_img)
        gif_name = "best.gif"
        mp4_name = None
        narrative_summary_name = None
        narrative_keyframes = {}
        trajectory_stats_name = None
        if frames:
            save_gif(self.run_dir / gif_name, frames, fps=8)
            try:
                mp4_name = "best.mp4"
                save_mp4(self.run_dir / mp4_name, frames, fps=8)
            except Exception:
                mp4_name = None
        if narrative_cfg.get("enabled", False) and frames:
            from core.logger import save_json

            narrative_result = score_narrative_trajectory(frames, narrative_cfg)
            keyframe_dir = self.run_dir / "phase_keyframes"
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            for idx, phase in enumerate(narrative_result["phases"], start=1):
                name = f"phase{idx}_{phase['name']}.png"
                save_image(keyframe_dir / name, frames[phase["frame_index"]])
                narrative_keyframes[phase["name"]] = str(Path("phase_keyframes") / name)
            narrative_summary = {
                "total_score": narrative_result["total_score"],
                "phase_order_valid": narrative_result["phase_order_valid"],
                "expected_component_sequence": narrative_result["expected_component_sequence"],
                "actual_component_sequence": narrative_result["actual_component_sequence"],
                "phases": [
                    {
                        "name": phase["name"],
                        "frame_index": phase["frame_index"],
                        "frame_range": phase["frame_range"],
                        "target_components": phase["target_components"],
                        "score": phase["score"],
                        "stats": phase["stats"],
                    }
                    for phase in narrative_result["phases"]
                ],
            }
            narrative_summary_name = "narrative_summary.json"
            trajectory_stats_name = "trajectory_morphology.json"
            save_json(self.run_dir / narrative_summary_name, narrative_summary)
            save_json(self.run_dir / trajectory_stats_name, narrative_result["frame_stats"])
        else:
            narrative_result = None

        return {
            "mode": "asal_target",
            "prompt": cfg["prompt"],
            "best_score": float(best_score),
            "best_theta": np.asarray(best).tolist(),
            "num_frames": len(frames),
            "foundation_model": cfg["foundation_model"]["name"],
            "substrate": cfg["substrate"]["name"],
            "morphology_judge": morphology_cfg["name"] if morphology_cfg and morphology_cfg.get("enabled", False) else None,
            "morphology_weight": morphology_weight,
            "search_iters": int(search["iters"]),
            "search_pop": int(search["pop"]),
            "search_keep": int(search["keep"]),
            "gif": gif_name if frames else None,
            "mp4": mp4_name,
            "narrative_enabled": bool(narrative_cfg.get("enabled", False)),
            "narrative_summary": narrative_summary_name,
            "trajectory_stats": trajectory_stats_name,
            "narrative_keyframes": narrative_keyframes,
            "narrative_score": float(narrative_result["total_score"]) if narrative_result else None,
            "narrative_phase_order_valid": bool(narrative_result["phase_order_valid"]) if narrative_result else None,
        }
