"""Single dedicated MLX worker thread (gemma_web crash fix 2026-10-05).

MLX 0.32 uses thread-local streams: a lazy array created on thread A must not be
evaluated on thread B (`There is no Stream(cpu, 0) in current thread`). All MLX
work (mlx-whisper load + transcribe, mel_filters warm-up) is submitted here so
create and eval happen on one owner thread.

Callers block on the returned Future. Compatible with the older MLX_LOCK: the
lock still serialises any leftover direct callers, and this worker is the
preferred path.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_SENTINEL = object()


class MLXWorker:
    """One daemon thread + FIFO queue. submit(fn) -> Future; call(fn) blocks."""

    def __init__(self, name: str = "mlx-worker") -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._started = False
        self._start_lock = threading.Lock()
        self._mel_warmed = False

    def start(self) -> None:
        with self._start_lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            fn, args, kwargs, fut = item
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 — deliver to caller
                fut.set_exception(exc)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
        self.start()
        fut: Future = Future()
        self._q.put((fn, args, kwargs, fut))
        return fut

    def call(self, fn: Callable[..., T], *args: Any, timeout: Optional[float] = None, **kwargs: Any) -> T:
        return self.submit(fn, *args, **kwargs).result(timeout=timeout)

    def warm_mel_filters(self, n_mels: int = 128) -> None:
        """Evaluate mel_filters once on this worker so later STT never races on the lru_cache."""
        if self._mel_warmed:
            return

        def _warm() -> None:
            import mlx.core as mx
            from mlx_whisper.audio import mel_filters

            mx.eval(mel_filters(n_mels))

        self.call(_warm)
        self._mel_warmed = True

    @property
    def alive(self) -> bool:
        return self._started and self._thread.is_alive()

    def queue_depth(self) -> int:
        return self._q.qsize()


_WORKER = MLXWorker()


def get_worker() -> MLXWorker:
    return _WORKER


def mlx_call(fn: Callable[..., T], *args: Any, timeout: Optional[float] = None, **kwargs: Any) -> T:
    """Run ``fn`` on the dedicated MLX thread; block until done."""
    return _WORKER.call(fn, *args, timeout=timeout, **kwargs)


def warm_mel_filters(n_mels: int = 128) -> None:
    _WORKER.warm_mel_filters(n_mels)
