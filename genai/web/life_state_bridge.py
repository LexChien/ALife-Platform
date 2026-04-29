"""
genai/web/life_state_bridge.py
Phase H1: Humanoid Foundation
Handles the bridge between ASAL live state and the Web session.
"""

from typing import Any, Dict
from pathlib import Path
import json

class LifeStateBridge:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def get_narrative_context(self, life_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts relevant metrics for LLM prompt injection.
        """
        if not life_snapshot:
            return {}
            
        return {
            "phase": life_snapshot.get("live_state", "idle"),
            "energy": life_snapshot.get("energy", 0.0),
            "num_components": life_snapshot.get("num_components", 0),
            "life_likeness": life_snapshot.get("life_likeness", 0.5)
        }

    def format_as_prompt_extra(self, context: Dict[str, Any]) -> str:
        """
        Formats the life context for inclusion in the system prompt.
        """
        if not context:
            return ""
            
        return (
            f"Current Physical State: {context['phase']}\n"
            f"Vitality Metrics: Energy={context['energy']:.1f}, "
            f"Complexity={context['num_components']}, "
            f"Life-Likeness={context['life_likeness']:.2f}"
        )
