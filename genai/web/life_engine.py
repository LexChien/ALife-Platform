import logging
import threading
import time
from pathlib import Path
import io
import base64
import torch

try:
    import numpy as np
    from PIL import Image
    HAS_VISUAL_DEPS = True
except ImportError:
    HAS_VISUAL_DEPS = False

from substrates.nca import NCA
from core.logger import iso_now, save_json
from foundation_models import foundation_models

logger = logging.getLogger(__name__)

class LiveLifeManager:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.run_dir / "best_frames"
        self.assets_dir.mkdir(exist_ok=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.substrate = NCA(device=device)
        self.theta = [0.0] * 5
        self._update_dna_signature()
        
        self.vlm = None
        if HAS_VISUAL_DEPS:
            try:
                self.vlm = foundation_models.create("openclip", device=device)
                logger.info("LiveLife VLM openclip ready")
            except Exception as e:
                logger.warning(f"LiveLife VLM unavailable (score may stay -1): {e}")
        
        self.target_text_emb = None
        self.state_to_phase = {"idle": "birth", "listening": "birth", "thinking": "split", "speaking": "fusion"}
        self.current_state = "idle"
        self.last_frame = None
        
        # P1 Metrics (Refined per Judgment)
        self.best_combined_score = -1.0
        self.current_combined_score = -1.0
        self.current_sem_score = 0.0
        self.current_penalty = 0.0
        self.target_energy = 5000.0
        
        # Checkpoint for Rollback
        self.checkpoint_weights = None
        self._save_checkpoint()

        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.evolution_log = []
        self.start_time = time.time()

    def _save_checkpoint(self):
        self.checkpoint_weights = [p.clone() for p in self.substrate.update_net.parameters()]

    def _apply_rollback(self, reason="Stability"):
        with torch.no_grad():
            for param, ckpt in zip(self.substrate.update_net.parameters(), self.checkpoint_weights):
                param.copy_(ckpt)
        self.substrate.reset()
        logger.warning(f"NCA ROLLBACK triggered by {reason}. Reverting to last best-stable state.")

    def _update_dna_signature(self):
        try:
            with torch.no_grad():
                convs = [m for m in self.substrate.update_net if isinstance(m, torch.nn.Conv2d)]
                if not convs:
                    return
                # Prefer last conv; detach because weights may require grad.
                w = convs[-1].weight.detach().flatten().cpu()
                vals = w[:5].tolist()
                # If still near-zero (untrained tiny slice), mix in earlier-layer signal.
                if max(abs(v) for v in vals) < 1e-8 and len(convs) > 1:
                    w0 = convs[0].weight.detach().flatten().cpu()
                    vals = w0[:5].tolist()
                self.theta = [round(float(x), 5) for x in vals]
        except Exception as e:
            logger.warning(f"DNA signature update failed: {e}")
            # keep previous theta rather than wiping to zeros

    def start(self):
        if self._thread is not None: return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def set_state(self, state: str, prompt: str | None = None):
        if state in self.state_to_phase:
            prev = self.current_state
            self.current_state = state
            if prompt:
                # Keep a short thought breadcrumb for the heartbeat log (thinking process trail).
                self.last_thought = (prompt[:160] + "…") if len(prompt) > 160 else prompt
            if prompt and self.vlm:
                try: self.target_text_emb = self.vlm.txt_embed(prompt)
                except Exception: pass
            # Flush immediately so short thinking windows are not missed by the 120-step sampler.
            if state != prev or prompt:
                try:
                    self._record_evolution_data(force=True)
                except Exception as e:
                    logger.error(f"Failed to flush evolution pulse on state={state}: {e}")

    def get_latest_frame_b64(self) -> str:
        if not HAS_VISUAL_DEPS: return ""
        with self.frame_lock:
            if self.last_frame is None: return ""
            buffered = io.BytesIO()
            self.last_frame.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _run_loop(self):
        step_count = 0
        while not self._stop_event.is_set():
            self.substrate.step(substeps=1)
            
            # P0 Stability Check: Rollback BEFORE the hard energy cap (20k)
            nca_stats = self.substrate.stats()
            if nca_stats["energy"] > 15000.0 or nca_stats["mass_ratio"] > 0.3:
                self._apply_rollback("Energy/Spatial Violation")

            if step_count % 60 == 0 and self.target_text_emb is not None and HAS_VISUAL_DEPS:
                self._perform_live_evolution()
            
            if step_count % 3 == 0 and HAS_VISUAL_DEPS:
                frame = self.substrate.render()
                if frame:
                    with self.frame_lock: self.last_frame = frame
            
            if step_count % 120 == 0:
                self._update_dna_signature()
                self._record_evolution_data()
            
            step_count += 1
            time.sleep(0.04)

    def _perform_live_evolution(self):
        if not HAS_VISUAL_DEPS or self.vlm is None: return
        
        original_weights = [p.clone() for p in self.substrate.update_net.parameters()]
        with torch.no_grad():
            for param in self.substrate.update_net.parameters():
                param.add_(torch.randn(param.size(), device=self.substrate.device) * 0.005)
        
        with self.frame_lock:
            if self.last_frame is None: return
            img = self.last_frame
            
        try:
            img_emb = self.vlm.img_embed(img)
            self.current_sem_score = float(np.dot(img_emb, self.target_text_emb))
            
            nca_stats = self.substrate.stats()
            # F-19 Refined: Energy Penalty + Spatial Penalty (to prevent blob)
            energy_penalty = 0.05 * abs(np.log10(nca_stats["energy"] + 1) - np.log10(self.target_energy + 1))
            spatial_penalty = 0.5 * max(0, nca_stats["mass_ratio"] - 0.15) # Penalize mass over 15%
            
            self.current_penalty = energy_penalty + spatial_penalty
            self.current_combined_score = self.current_sem_score - self.current_penalty
            
            if self.current_combined_score > self.best_combined_score:
                self.best_combined_score = self.current_combined_score
                self._save_checkpoint()
                img.save(self.assets_dir / f"best_{int(time.time())}.png")
                logger.info(f"NCA Improvement: Combined={self.current_combined_score:.4f} (Sem={self.current_sem_score:.4f})")
            else:
                with torch.no_grad():
                    for param, old in zip(self.substrate.update_net.parameters(), original_weights):
                        param.copy_(old)
        except Exception as e:
            logger.error(f"Evolution step error: {e}")

    def _record_evolution_data(self, force: bool = False):
        # Always allow forced pulses (state transitions) even before first frame.
        nca_stats = {}
        if HAS_VISUAL_DEPS:
            with self.frame_lock:
                if self.last_frame is None and not force:
                    return
                try:
                    nca_stats = self.substrate.stats()
                except Exception:
                    nca_stats = {}
        elif not force:
            return

        self._update_dna_signature()
        score = round(self.best_combined_score, 4)
        entry = {
            "timestamp": iso_now(),
            "elapsed": round(time.time() - self.start_time, 2),
            "state": self.current_state,
            "phase": self.state_to_phase.get(self.current_state, "N/A"),
            "best_score": score,  # viz_life_history SCORE column
            "current_combined": round(self.current_combined_score, 4),
            "best_combined": score,
            "semantic_score": round(self.current_sem_score, 4),
            "energy_penalty": round(self.current_penalty, 4),
            "dna": self.theta,
            "morphology": {
                **nca_stats,
                # viz expects Comp/Area
                "num_components": nca_stats.get("num_components", nca_stats.get("components", 1)),
                "largest_area": nca_stats.get("largest_area", nca_stats.get("active_mass", nca_stats.get("mass", 0))),
            },
            "thought": getattr(self, "last_thought", "") if self.current_state == "thinking" else "",
        }
        self.evolution_log.append(entry)
        # Persist start_time once (first pulse), not overwritten each write.
        if not hasattr(self, "_history_start_iso"):
            self._history_start_iso = iso_now()
        save_json(
            self.run_dir / "live_evolution.json",
            {
                "system": "live_asal",
                "start_time": self._history_start_iso,
                "evolution_history": self.evolution_log,
            },
        )

    def stop(self):
        self._stop_event.set()
        if self._thread: self._thread.join()
