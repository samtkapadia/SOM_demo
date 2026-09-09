import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kohonen.mosaic import atlas_to_png_bytes
from kohonen.som import SOM


WIDTH = 16
HEIGHT = 16
N_ITER = 100
SEED = 7

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path.home() / ".cache" / "som-atlas" / "embeddings.npz"
OUTPUT_PATH = ROOT / "outputs" / "atlas_mosaic.png"


def main():
    data = np.load(DATA_PATH, allow_pickle=False)
    X = data["X"]

    som = SOM(WIDTH, HEIGHT, N_ITER, seed=SEED).train(X)
    png = atlas_to_png_bytes(som.weights, X, data["thumbs"])

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_bytes(png)
    print(f"quantization error: {som.quantization_error(X):.4f}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
