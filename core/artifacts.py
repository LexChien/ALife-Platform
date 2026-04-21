from pathlib import Path
from PIL import Image
import numpy as np

def save_image(path, arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    Image.fromarray(arr).save(Path(path))
