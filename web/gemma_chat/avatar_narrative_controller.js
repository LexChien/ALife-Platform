/**
 * web/gemma_chat/avatar_narrative_controller.js
 * Phase H1: Humanoid Foundation
 * Handles mapping from ASAL phases to Avatar visual states.
 */

class AvatarNarrativeController {
    constructor() {
        this.phaseToState = {
            "birth": "idle",
            "split": "thinking",
            "fusion": "speaking",
            "idle": "idle",
            "listening": "listening"
        };
        this.currentPhase = "idle";
    }

    /**
     * Updates the avatar visual intensity based on ASAL metrics.
     * @param {Object} metrics - ASAL metrics (energy, components, etc.)
     * @returns {number} Intensity value (0.0 to 1.0)
     */
    calculateIntensity(metrics) {
        if (!metrics) return 0.5;
        // Intensity proportional to energy and component count as defined in PLAN 33
        const energyNorm = Math.min(metrics.energy / 20000.0, 1.0);
        const compNorm = Math.min(metrics.num_components / 10.0, 1.0);
        return (energyNorm + compNorm) / 2.0;
    }

    /**
     * Maps ASAL phase to Avatar state.
     * @param {string} phase - Current ASAL phase
     * @returns {string} Target Avatar state
     */
    getAvatarState(phase) {
        this.currentPhase = phase || "idle";
        return this.phaseToState[this.currentPhase.toLowerCase()] || "idle";
    }
}

// Global instance
const avatarNarrative = new AvatarNarrativeController();
