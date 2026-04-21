import numpy as np
from foundation_models import foundation_models
from core.artifacts import save_image
from core.logger import save_json
from .scores import supervised_target_score
from .search.optim import evo_search
from .substrates.reaction_diffusion import ReactionDiffusion

class ASALEngine:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir

    def run(self):
        cfg = self.config
        fm = foundation_models.create(cfg["foundation_model"]["name"], **cfg["foundation_model"].get("params", {}))
        substrate = ReactionDiffusion(**cfg["substrate"].get("params", {}))
        txt_emb = fm.txt_embed(cfg["prompt"])

        search = cfg["search"]
        low = np.array(search["theta_low"], dtype=float)
        high = np.array(search["theta_high"], dtype=float)
        rs = np.random.RandomState(search.get("seed", 0))
        init = [rs.uniform(low, high) for _ in range(search["keep"])]

        def eval_theta(theta):
            substrate.reset(theta)
            img_embs = []
            for _ in range(cfg["runtime"]["steps"]):
                substrate.step(cfg["runtime"]["substeps"])
            img = substrate.render()
            img_embs.append(fm.img_embed(img))
            return supervised_target_score(img_embs, txt_emb)

        pool, scores, best, best_score = evo_search(
            init, eval_theta,
            iters=search["iters"],
            pop=search["pop"],
            keep=search["keep"],
            sigma=search["sigma"],
            bounds=(low, high),
        )

        substrate.reset(best)
        for _ in range(cfg["runtime"]["steps"]):
            substrate.step(cfg["runtime"]["substeps"])
        best_img = np.asarray(substrate.render())
        save_image(self.run_dir / "best.png", best_img)
        save_json(self.run_dir / "summary.json", {
            "mode": "asal_target",
            "prompt": cfg["prompt"],
            "best_score": best_score,
            "best_theta": np.asarray(best).tolist(),
        })
