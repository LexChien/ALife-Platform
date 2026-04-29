#!/usr/bin/env python3
"""
core/co_evolution.py
Phase H2: Life-Mind Co-Evolution Closed Loop
嚴格實作 30 文件公式 + 近期 NCA rollback 機制
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from core.logger import get_logger
from core.artifacts import save_json

logger = get_logger(__name__)


@dataclass
class CoEvolutionMetrics:
    life_likeness: float          # L 值 (0.3~0.7 為理想區間)
    semantic_score: float
    energy_penalty: float
    combined_score: float
    persona_drift: float
    components: int
    mass: float
    energy: float
    timestamp: float = field(default_factory=time.time)


class LifeMindFeedbackLoop:
    """
    心身共同演化控制器
    觸發條件：
      - L 偏離 0.5 超過 50 steps → 自動 LoRA 微調
      - L 穩定在 0.3~0.7 超過 100 steps → 獎勵當前配置
    """

    def __init__(
        self,
        config: Dict[str, Any],
        artifact_dir: Path,
        clone_evaluator: Any,          # digital_clone.consistency.evaluator
        llm_adapter: Any,              # genai.llm.adapter
        rollback_fn: Optional[callable] = None,
    ):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.clone_evaluator = clone_evaluator
        self.llm_adapter = llm_adapter
        self.rollback_fn = rollback_fn

        self.history: list[CoEvolutionMetrics] = []
        self.consecutive_off_target = 0
        self.consecutive_stable = 0
        self.last_lora_version = 0

        self.ideal_min = self.config.get("ideal_life_likeness_range", [0.3, 0.7])[0]
        self.ideal_max = self.config.get("ideal_life_likeness_range", [0.3, 0.7])[1]
        self.trigger_threshold = self.config.get("trigger_off_target_steps", 50)
        self.reward_threshold = self.config.get("reward_stable_steps", 100)

    def step(
        self,
        asal_metrics: Dict[str, Any],
        clone_state: Dict[str, Any],
        genai_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        每步驟呼叫一次
        返回: {"action": "continue" | "tune_lora" | "reward", "metrics": ...}
        """
        metrics = CoEvolutionMetrics(
            life_likeness=asal_metrics.get("life_likeness", 0.5),
            semantic_score=asal_metrics.get("semantic_score", 0.0),
            energy_penalty=asal_metrics.get("energy_penalty", 0.0),
            combined_score=asal_metrics.get("combined_score", 0.0),
            persona_drift=clone_state.get("persona_drift", 0.0),
            components=asal_metrics.get("components", 1),
            mass=asal_metrics.get("mass", 0.0),
            energy=asal_metrics.get("energy", 0.0),
        )

        self.history.append(metrics)
        if len(self.history) > 200:
            self.history.pop(0)

        # 更新連續計數器
        if not (self.ideal_min <= metrics.life_likeness <= self.ideal_max):
            self.consecutive_off_target += 1
            self.consecutive_stable = 0
        else:
            self.consecutive_stable += 1
            self.consecutive_off_target = 0

        action = "continue"

        # === 觸發 LoRA 微調 ===
        if self.consecutive_off_target >= self.trigger_threshold:
            action = "tune_lora"
            dataset = self._generate_tuning_dataset(asal_metrics, clone_state, genai_output)
            self._trigger_lora_tuning(dataset)
            self.consecutive_off_target = 0
            logger.warning(f"[CoEvolution] Triggered LoRA tuning. L={metrics.life_likeness:.4f}")

        # === 獎勵當前配置 ===
        elif self.consecutive_stable >= self.reward_threshold:
            action = "reward"
            self._reduce_mutation_rate()
            logger.info(f"[CoEvolution] Rewarding stable configuration. L={metrics.life_likeness:.4f}")

        # 記錄 Artifact
        self._save_co_evolution_log(metrics, action)

        # 整合 rollback（若 energy 異常）
        if metrics.energy > 30000 and self.rollback_fn:
            self.rollback_fn()
            logger.warning("[CoEvolution] Energy anomaly detected. Rollback triggered.")

        return {
            "action": action,
            "metrics": metrics,
            "consecutive_off_target": self.consecutive_off_target,
            "consecutive_stable": self.consecutive_stable,
        }

    def _generate_tuning_dataset(
        self,
        asal_metrics: Dict[str, Any],
        clone_state: Dict[str, Any],
        genai_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """產生 LoRA 微調 dataset（簡化版，實際應擴充）"""
        return {
            "life_phase": asal_metrics.get("phase", "unknown"),
            "life_likeness": asal_metrics.get("life_likeness"),
            "persona": clone_state.get("persona"),
            "user_input": genai_output.get("input", ""),
            "ideal_response": genai_output.get("output", ""),
            "timestamp": time.time(),
        }

    def _trigger_lora_tuning(self, dataset: Dict[str, Any]):
        """呼叫外部 LoRA 微調流程"""
        self.last_lora_version += 1
        save_json(
            self.artifact_dir / f"lora_tuning_request_v{self.last_lora_version}.json",
            dataset,
        )
        # TODO: 實際呼叫 training/lora/train_clone_lora.py

    def _reduce_mutation_rate(self):
        """降低 ASAL mutation rate"""
        # TODO: 與 life_engine 介面對接
        pass

    def _save_co_evolution_log(self, metrics: CoEvolutionMetrics, action: str):
        log_path = self.artifact_dir / "co_evolution_log.jsonl"
        entry = {
            "ts": metrics.timestamp,
            "life_likeness": round(metrics.life_likeness, 4),
            "combined_score": round(metrics.combined_score, 4),
            "action": action,
            "lora_version": self.last_lora_version,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
