"""
Dataset loading for error mitigation experiments.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_moons_dataset(n_samples=200, noise=0.15, seed=42):
    """make_moons scaled to [0, pi]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def create_meshgrid(X, resolution=30):
    """Create meshgrid for decision boundary plots."""
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]
