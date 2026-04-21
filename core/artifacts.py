from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
from PIL import Image
import numpy as np
import imageio.v2 as imageio

def save_image(path, arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    Image.fromarray(arr).save(Path(path))


def _as_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    return arr


def save_gif(path, frames, fps=8):
    frames = [_as_uint8(frame) for frame in frames]
    imageio.mimsave(Path(path), frames, format="GIF", duration=1 / max(fps, 1))


def save_mp4(path, frames, fps=8):
    frames = [_as_uint8(frame) for frame in frames]
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        imageio.mimsave(Path(path), frames, format="mp4", fps=fps)
