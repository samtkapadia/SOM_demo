import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kohonen import SOM
from kohonen.viz import save_rgb

n_samples = 10
n_features = 3
width = 100
height = 100
n_iter = 1000
seed = 7

out_dir = Path(__file__).resolve().parents[1] / "outputs"


def main():
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features))

    som = SOM(width, height, n_iter, seed=seed)
    som.train(X)

    out_dir.mkdir(exist_ok=True)
    som.save(out_dir / "som.pkl")
    print(f"quantization error: {som.quantization_error(X):.4f}")

    if n_features == 3:
        save_rgb(som.weights, out_dir / "som.png")


if __name__ == "__main__":
    main()
