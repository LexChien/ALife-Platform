"""Unit tests for the dedicated MLX worker queue (no MLX required)."""
from __future__ import annotations

import threading
import time
import unittest

from genai.web.mlx_worker import MLXWorker


class MLXWorkerTests(unittest.TestCase):
    def test_serialises_callables_on_one_thread(self):
        w = MLXWorker(name="mlx-worker-test")
        seen_threads: list[str] = []
        concurrent = {"n": 0, "max": 0}
        lock = threading.Lock()

        def job(tag: str) -> str:
            with lock:
                concurrent["n"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["n"])
            seen_threads.append(threading.current_thread().name)
            time.sleep(0.03)
            with lock:
                concurrent["n"] -= 1
            return tag

        futs = [w.submit(job, f"t{i}") for i in range(4)]
        results = [f.result(timeout=5) for f in futs]
        self.assertEqual(results, ["t0", "t1", "t2", "t3"])
        self.assertEqual(concurrent["max"], 1)  # never overlapped
        self.assertEqual(set(seen_threads), {"mlx-worker-test"})

    def test_exceptions_propagate(self):
        w = MLXWorker(name="mlx-worker-exc")

        def boom():
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            w.call(boom)


if __name__ == "__main__":
    unittest.main()
