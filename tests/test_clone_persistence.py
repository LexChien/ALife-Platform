import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("chromadb"), "requires real ChromaDB; run with system python3")
class TestCrossProcessPersistence(unittest.TestCase):
    def test_restart_rename_isolation_and_idempotent_profiles(self):
        worker = Path(__file__).parent / "fixtures" / "memory_process.py"
        with tempfile.TemporaryDirectory() as database:
            env = {**os.environ, "ANONYMIZED_TELEMETRY": "False", "PYTHONDONTWRITEBYTECODE": "1"}
            outputs = []
            for action in ("write", "read"):
                proc = subprocess.run([sys.executable, str(worker), action, database],
                                      check=True, capture_output=True, text=True, env=env, timeout=60)
                outputs.append(json.loads(proc.stdout))
        before, after = outputs
        self.assertEqual(len({row["collection"] for row in after}), 5)
        for index, (written, loaded) in enumerate(zip(before, after)):
            self.assertEqual(written["collection"], loaded["collection"])
            self.assertEqual(loaded["current_items"], 0)
            self.assertEqual(loaded["count"], 2)
            self.assertEqual({item["content"] for item in loaded["memories"]},
                             {f"profile-{index}", f"private-memory-{index}"})
            profile = next(item for item in loaded["memories"] if item["kind"] == "profile_fact")
            self.assertEqual(profile["role"], "system")


if __name__ == "__main__":
    unittest.main()
