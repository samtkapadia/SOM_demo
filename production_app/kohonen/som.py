"""Vectorised Kohonen map — same update as train_vectorised in the notebook."""

import pickle
from pathlib import Path

import numpy as np


def find_bmu(weights, x):
    """
    Method to find the BMU for the input vector.
    """
    d = np.linalg.norm(weights - x, axis=2)
    return np.unravel_index(np.argmin(d), d.shape)


def influence(rows, cols, bmu_row, bmu_col, sigma_t, radius_mask):
    """
    Method to calculate the influence of the BMU on the SOM.
    """
    dist = np.hypot(rows - bmu_row, cols - bmu_col)
    theta = np.exp(-(dist**2) / (2 * sigma_t**2))
    if radius_mask:
        theta = theta * (dist <= sigma_t)
    return theta


class SOM:
    """
    Class for the Self-Organising Map (SOM).
    """
    def __init__(self, width, height, n_iter, lr0=0.1, sigma0=None, seed=None, radius_mask=False):
        self.width = width
        self.height = height
        self.n_iter = n_iter
        self.lr0 = lr0
        self.sigma0 = sigma0 if sigma0 is not None else max(width, height) / 2
        self.seed = seed
        self.radius_mask = radius_mask
        self.weights = None

    def init_weights(self, n_features):
        """
        Method to draw the untrained codebook from the seed.
        """
        rng = np.random.default_rng(self.seed)
        self.weights = rng.random((self.height, self.width, int(n_features)))
        return self

    def train(self, X):
        """
        Method to train the SOM on the input data.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected (n_samples, n_features), got {X.shape}")

        # schedule uses log(sigma0); a 2x2 grid with the default radius hits 1 and blows up
        if self.sigma0 <= 1:
            raise ValueError(f"sigma0 must be > 1, got {self.sigma0}")

        self.init_weights(X.shape[1])
        W = self.weights
        lam = self.n_iter / np.log(self.sigma0)
        rows, cols = np.indices((self.height, self.width))

        for t in range(self.n_iter):
            sigma_t = self.sigma0 * np.exp(-t / lam)
            lr_t = self.lr0 * np.exp(-t / lam)
            for i in range(len(X)):
                x = X[i].reshape(1, -1)
                br, bc = find_bmu(W, x)
                theta = influence(rows, cols, br, bc, sigma_t, self.radius_mask)
                W += lr_t * theta[..., None] * (x - W)

        self.weights = W
        return self

    def bmu(self, X):
        """
        Method to return the coordinates of the BMU for each input vector.
        """
        X = np.asarray(X, dtype=np.float64)
        coords = np.empty((len(X), 2), dtype=int)
        for i in range(len(X)):
            coords[i] = find_bmu(self.weights, X[i])
        return coords

    def quantization_error(self, X):
        """
        Method to return the quantization error for the SOM.
        """
        X = np.asarray(X, dtype=np.float64)
        idx = self.bmu(X)
        return float(np.linalg.norm(X - self.weights[idx[:, 0], idx[:, 1]], axis=1).mean())

    def save(self, path):
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as f:
            return pickle.load(f)
