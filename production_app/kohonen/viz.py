import matplotlib.pyplot as plt
import numpy as np


def save_rgb(weights, path):
    """
    Function for saving the SOM weights as a PNG image (only works for 3-D feature maps treated as RGB).
    """
    if weights.shape[-1] != 3:
        raise ValueError(f"need 3 features for RGB, got {weights.shape}")
    plt.imsave(path, np.clip(weights, 0, 1))
