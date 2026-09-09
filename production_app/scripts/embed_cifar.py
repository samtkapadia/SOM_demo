"""Build the small, fixed CIFAR-10 embedding dataset used by the photo atlas.

This is an offline preparation step. PyTorch and scikit-learn are deliberately
not installed in Lambda.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


N_PER_CLASS = 25
N_COMPONENTS = 128
THUMB_SIZE = 48
SEED = 7

CACHE_DIR = Path.home() / ".cache" / "som-atlas"
DOWNLOAD_DIR = CACHE_DIR / "cifar10"
OUTPUT_PATH = CACHE_DIR / "embeddings.npz"


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def select_balanced_indices(targets):
    rng = np.random.default_rng(SEED)
    targets = np.asarray(targets)
    selected = []
    for class_id in range(10):
        candidates = np.flatnonzero(targets == class_id)
        selected.extend(rng.choice(candidates, N_PER_CLASS, replace=False))
    return np.asarray(selected)


def main():
    dataset = CIFAR10(DOWNLOAD_DIR, train=False, download=True)
    indices = select_balanced_indices(dataset.targets)

    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    images = [dataset[int(i)][0] for i in indices]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = torch.stack([transform(image) for image in images[start : start + 32]])
            batches.append(model(batch).cpu().numpy())

    features_512 = l2_normalize(np.concatenate(batches))
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    embeddings = l2_normalize(pca.fit_transform(features_512)).astype(np.float32)

    thumbs = np.stack(
        [
            np.asarray(image.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.NEAREST))
            for image in images
        ]
    ).astype(np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)[indices]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X=embeddings,
        labels=labels,
        thumbs=thumbs,
        class_names=np.asarray(dataset.classes),
        encoder=np.asarray("ImageNet ResNet-18 avgpool; PCA 512 -> 128"),
    )
    print(f"wrote {OUTPUT_PATH}")
    print(f"X: {embeddings.shape}, thumbs: {thumbs.shape}")
    print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
