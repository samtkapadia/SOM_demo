import io

import numpy as np
from PIL import Image


def weights_to_png_bytes(weights) -> bytes:
    if weights.ndim != 3 or weights.shape[-1] != 3:
        raise ValueError(f"need (height, width, 3), got {weights.shape}")
    arr = (np.clip(weights, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
