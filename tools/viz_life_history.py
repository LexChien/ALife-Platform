import json
import sys
import time
from pathlib import Path
from datetime import datetime

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
            # If file is being written to, wait a bit
            time.sleep(0.5)
            continue

        history = data.get("evolution_history", [])
        if not history:
            if not follow:
                print("No evolution data found.")
                return
            time.sleep(1)
            continue

        # Print header only the first time
        if last_entry_count == 0:
            print("\n" + "="*80)
            print(f" ARTIFICIAL LIFE EVOLUTIONARY HISTORY & HEARTBEAT LOG")
            print(f" System: {data.get('system', 'unknown')} | Started at: {data.get('start_time', 'N/A')}")
            print("="*80 + "\n")
            print(f"{'TIME (s)':<10} | {'STATE':<10} | {'PHASE':<8} | {'SCORE':<8} | {'MORPHOLOGY (Comp/Area)':<20} | {'DNA (Theta) Snippet'}")
            print("-" * 110)

        # Print only new entries
        new_entries = history[last_entry_count:]
        last_score = -1.0 if last_entry_count == 0 else history[last_entry_count-1].get("best_score", 0.0)

        for entry in new_entries:
            elapsed = entry.get("elapsed", 0.0)
            state = entry.get("state", "idle")
            phase = entry.get("phase", "N/A")
            score = entry.get("best_score", 0.0)
            morph = entry.get("morphology", {})
            dna = entry.get("dna", [])
            
            # Highlight evolution progress
            score_str = f"{score:.4f}"
            if score > last_score and last_score != -1.0:
                score_str = f"\033[92m{score_str} ↑\033[0m" # Green for improvement
            last_score = max(score, last_score)

            morph_str = f"{morph.get('num_components', 0)} / {int(morph.get('largest_area', 0))}"
            dna_snippet = ", ".join([f"{v:.2f}" for v in dna[:3]]) + "..."

            print(f"{elapsed:<10.1f} | {state:<10} | {phase:<8} | {score_str:<8} | {morph_str:<20} | [{dna_snippet}]")

        last_entry_count = len(history)

        if not follow:
            print("\n" + "="*80)
            print(f" SUMMARY: Total Pulses: {len(history)} | Peak Semantic Score: {last_score:.4f}")
            print("="*80 + "\n")
            break
        
        time.sleep(2) # Wait for next update

if __name__ == "__main__":
    follow_mode = False
    args = sys.argv[1:]
    
    if "-f" in args:
        follow_mode = True
        args.remove("-f")
    
    path = args[0] if args else "runs/chat_gemma_web/latest/live_engine/live_evolution.json"
    visualize_history(path, follow=follow_mode)
