import json
import tempfile
import unittest
from pathlib import Path

from genai.web.life_state import ASALProgressIndex


class ASALProgressIndexTests(unittest.TestCase):
    def test_snapshot_reads_latest_asal_run_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "asal" / "20260424-195441"
            run_dir.mkdir(parents=True)
            (run_dir / "best.gif").write_bytes(b"GIF89a")
            keyframe_dir = run_dir / "phase_keyframes"
            keyframe_dir.mkdir()
            (keyframe_dir / "phase1_birth.png").write_bytes(b"PNG")
            (run_dir / "narrative_summary.json").write_text(
                json.dumps(
                    {
                        "total_score": 0.78,
                        "phase_order_valid": True,
                        "actual_component_sequence": [1, 2, 1],
                        "expected_component_sequence": [1, 2, 1],
                        "phases": [
                            {"name": "birth", "score": 0.7, "target_components": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "mode": "asal_target",
                        "config_path": "configs/asal/target_cell_fusion_narrative.yaml",
                        "completed_at": "2026-04-24T20:00:29+08:00",
                        "artifacts": {
                            "gif": "best.gif",
                            "narrative_summary": "narrative_summary.json",
                        },
                        "metrics": {"best_score": 0.83},
                        "details": {
                            "substrate": "boids",
                            "foundation_model": "morphology_judge_stub",
                            "narrative_keyframes": {
                                "birth": "phase_keyframes/phase1_birth.png",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            index = ASALProgressIndex(root=root)
            snapshot = index.snapshot()

            self.assertEqual(snapshot["latest_run"]["run_id"], "20260424-195441")
            self.assertEqual(snapshot["latest_run"]["details"]["actual_component_sequence"], [1, 2, 1])
            self.assertEqual(
                snapshot["latest_run"]["details"]["phase_scores"][0]["keyframe_url"],
                "/artifacts/asal/20260424-195441/phase_keyframes/phase1_birth.png",
            )
            self.assertIn("/artifacts/asal/20260424-195441/best.gif", snapshot["latest_run"]["asset_urls"]["gif"])
            artifact = index.resolve_artifact("20260424-195441", "best.gif")
            self.assertEqual(artifact.read_bytes(), b"GIF89a")

    def test_context_text_warns_against_overclaiming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = ASALProgressIndex(root=Path(tmpdir))
            text = index.context_text()
            self.assertIn("不得誇大", text)
            self.assertIn("雛形", text)


if __name__ == "__main__":
    unittest.main()
