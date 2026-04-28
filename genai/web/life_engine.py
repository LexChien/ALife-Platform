import logging
import threading
import time
from pathlib import Path
import io
import base64

try:
    import numpy as np
    from PIL import Image
    HAS_VISUAL_DEPS = True
except ImportError:
    HAS_VISUAL_DEPS = False

from substrates.boids import Boids
from core.logger import iso_now
from foundation_models import foundation_models

logger = logging.getLogger(__name__)

class LiveLifeManager:
    """
    Manages a real-time running ASAL substrate that responds to GenAI states.
    Includes a live evolution loop triggered by interactions.
    """
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.substrate_cfg = config.get("substrate", {"name": "boids", "params": {}})
        # Ensure narrative controller is enabled for live phase switching
        params = self.substrate_cfg.get("params", {}).copy()
        if "narrative_controller" not in params:
            params["narrative_controller"] = {
                "enabled": True,
                "base_center_pull": 0.015,
                "base_velocity_damping": 0.08,
                "phases": [
                    {"name": "birth", "center_pull": 0.04, "damping": 0.15, "separation_scale": 0.2, "cohesion_scale": 1.5},
                    {"name": "split", "split_push": 0.05, "cohort_pull": 0.03, "split_offset": 25.0, "separation_scale": 1.0},
                    {"name": "fusion", "center_pull": 0.05, "damping": 0.18, "separation_scale": 0.1, "cohesion_scale": 1.8}
                ]
            }
        self.substrate = Boids(**params)
        self.theta = np.array([1.0, 1.0, 1.0, 5.0, 2.0]) if HAS_VISUAL_DEPS else [1.0]*5
        
        # Perception: AI Judge
        self.vlm = None
        if HAS_VISUAL_DEPS:
            try:
                self.vlm = foundation_models.create("openclip", device="cpu")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCLIP for LiveLife: {e}. Using random scoring fallback.")
        
        self.target_text_emb = None
        
        # Mapping GenAI states to Narrative phases
        self.state_to_phase = {
            "idle": "birth",
            "listening": "birth",
            "thinking": "split",
            "speaking": "fusion"
        }
        
        self.current_state = "idle"
        self.last_frame = None
        self.best_score = -1.0
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Evolution / Growth stats
        self.evolution_log = []
        self.start_time = time.time()

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("LiveLifeManager started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        logger.info("LiveLifeManager stopped.")

    def set_state(self, state: str, prompt: str | None = None):
        """Sets the cognitive state which influences the physical morphology."""
        if state in self.state_to_phase:
            self.current_state = state
            phase_name = self.state_to_phase[state]
            self.substrate.set_active_phase(phase_name)
            
            # If we have a new prompt, update the evolutionary target
            if prompt and self.vlm:
                try:
                    self.target_text_emb = self.vlm.txt_embed(prompt)
                    logger.info(f"Live Evolution target updated to: {prompt}")
                except Exception as e:
                    logger.error(f"Failed to update target embedding: {e}")
                    
            logger.info(f"LiveLifeManager state changed to: {state} (Phase: {phase_name})")

    def get_latest_frame_b64(self) -> str:
        if not HAS_VISUAL_DEPS:
            return ""
        with self.frame_lock:
            if self.last_frame is None:
                return ""
            buffered = io.BytesIO()
            self.last_frame.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _run_loop(self):
        step_count = 0
        while not self._stop_event.is_set():
            # 1. Occasional Mutation / Selection (Evolution)
            if step_count % 60 == 0 and self.target_text_emb is not None and HAS_VISUAL_DEPS:
                self._perform_live_evolution()

            # 2. Physics step
            self.substrate.step(substeps=1)
            
            # 3. Render and update last frame
            if step_count % 3 == 0 and HAS_VISUAL_DEPS:
                frame = self.substrate.render()
                with self.frame_lock:
                    self.last_frame = frame
            
            # 4. Occasional logging of "Evolution Data"
            if step_count % 120 == 0:
                self._record_evolution_data()
            
            step_count += 1
            time.sleep(0.04) # ~25 FPS simulation

    def _perform_live_evolution(self):
        """Slightly mutate theta and check if it improves visual match to current context."""
        if not HAS_VISUAL_DEPS or self.vlm is None:
            return
            
        sigma = 0.08
        mutation = np.random.randn(len(self.theta)) * sigma
        candidate_theta = np.clip(self.theta + mutation, 0.1, 12.0)
        
        with self.frame_lock:
            if self.last_frame is None: return
            img = self.last_frame
            
        try:
            img_emb = self.vlm.img_embed(img)
            score = float(np.dot(img_emb, self.target_text_emb))
            
            if score > self.best_score:
                self.best_score = score
                self.theta = candidate_theta
                # Directly update substrate params
                self.substrate.base_separation = self.theta[0]
                self.substrate.base_alignment = self.theta[1]
                self.substrate.base_cohesion = self.theta[2]
                self.substrate.base_speed = self.theta[3]
                self.substrate.base_view_radius = self.theta[4]
                logger.info(f"Live Evolution: Mutation Accepted! New Score: {score:.4f}")
        except Exception:
            pass

    def _record_evolution_data(self):
        if not HAS_VISUAL_DEPS:
            return
            
        from research.asal_engine.morphology import analyze_frame
        
        with self.frame_lock:
            if self.last_frame is None:
                return
            stats = analyze_frame(np.array(self.last_frame))
            
        entry = {
            "timestamp": iso_now(),
            "elapsed": round(time.time() - self.start_time, 2),
            "state": self.current_state,
            "phase": self.state_to_phase.get(self.current_state),
            "best_score": round(self.best_score, 4),
            "morphology": {
                "num_components": stats["num_components"],
                "dominant_num": stats["dominant_num_components"],
                "largest_area": stats["largest_area"],
                "mass_ratio": stats["dominant_mass_ratio"]
            },
            "dna": self.theta.tolist()
        }
        self.evolution_log.append(entry)
        
        # Save to runs directory periodically
        log_path = self.run_dir / "live_evolution.json"
        from core.logger import save_json
        save_json(log_path, {
            "system": "live_asal",
            "start_time": iso_now(),
            "evolution_history": self.evolution_log
        })
