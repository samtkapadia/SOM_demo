import io

import numpy as np
from PIL import Image


def nearest_image_indices(weights, embeddings):
    """Index of the nearest training image for each SOM cell."""
    cells = weights.reshape(-1, weights.shape[-1])
    cell_norm = np.sum(cells**2, axis=1, keepdims=True)
    embedding_norm = np.sum(embeddings**2, axis=1)
    distances = cell_norm + embedding_norm - 2 * cells @ embeddings.T
    return np.argmin(distances, axis=1).reshape(weights.shape[:2])


def atlas_to_png_bytes(weights, embeddings, thumbs):
    nearest = nearest_image_indices(weights, embeddings)
    height, width = nearest.shape
    thumb_height, thumb_width = thumbs.shape[1:3]
    canvas = np.empty(
        (height * thumb_height, width * thumb_width, 3),
        dtype=np.uint8,
    )

    for row in range(height):
        for col in range(width):
            top = row * thumb_height
            left = col * thumb_width
            canvas[top : top + thumb_height, left : left + thumb_width] = thumbs[
                nearest[row, col]
            ]

    buf = io.BytesIO()
    Image.fromarray(canvas, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
