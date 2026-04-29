import json
import sys
import time
from pathlib import Path

def visualize_history(json_path, follow=False):
    if not Path(json_path).exists():
        print(f"Error: File {json_path} not found.")
        return

    last_entry_count = 0
    while True:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(0.5); continue

        history = data.get("evolution_history", [])
        system_type = data.get("system", "unknown")
        if not history:
            if not follow: return
            time.sleep(1); continue

        if last_entry_count == 0:
            print("\n" + "="*190)
            print(f" ARTIFICIAL LIFE MONITORING | System: {system_type} | Started at: {data.get('start_time', 'N/A')}")
            print("="*190 + "\n")
            print(f"{'TIME (s)':<8} | {'STATE':<10} | {'PHASE':<8} | {'SEMANTIC':<8} | {'PENALTY':<8} | {'CURR_COMB':<10} | {'BEST_COMB':<10} | {'COMP/MASS':<12} | {'ENERGY':<10} | {'STABILITY':<12}")
            print("-" * 190)

        new_entries = history[last_entry_count:]
        
        for entry in new_entries:
            elapsed = entry.get("elapsed", 0.0)
            state = entry.get("state", "idle")
            phase = entry.get("phase", "N/A")
            
            sem_score = entry.get("semantic_score", 0.0)
            penalty = entry.get("energy_penalty", 0.0)
            curr_comb = entry.get("current_combined", 0.0)
            best_comb = entry.get("best_combined", 0.0)
            
            comb_str = f"{curr_comb:.4f}"
            if curr_comb >= best_comb and best_comb > -1.0:
                comb_str = f"\033[92m{comb_str} *\033[0m" # Highlight best

            morph = entry.get("morphology", {})
            comp_mass = f"{morph.get('num_components', 0)} / {int(morph.get('active_mass', 0))}"
            energy = f"{morph.get('energy', 0.0):.1f}"
            
            if morph.get("clamped", False):
                status = "\033[91mCLAMPED\033[0m"
            elif morph.get("is_stable", True):
                status = "\033[94mSTABLE\033[0m"
            else:
                status = "\033[93mDRIFTING\033[0m"
            
            print(f"{elapsed:<8.1f} | {state:<10} | {phase:<8} | {sem_score:<8.4f} | {penalty:<8.4f} | {comb_str:<10} | {best_comb:<10.4f} | {comp_mass:<12} | {energy:<10} | {status:<21}")

        last_entry_count = len(history)
        if not follow: break
        time.sleep(1.5)

if __name__ == "__main__":
    follow_mode = "-f" in sys.argv
    path = [a for a in sys.argv[1:] if a != "-f"][0] if len(sys.argv) > 1 else "runs/chat_gemma_web/latest/live_engine/live_evolution.json"
    visualize_history(path, follow=follow_mode)
