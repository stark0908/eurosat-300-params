"""Feature preprocessing transformations for EuroSAT parameter minimization.

Both transformations are fit STRICTLY on the training split to prevent data leakage.
Neither transformation introduces any learned parameters into the deployed classifier head.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler


class StandardScalerTransform:
    """Standard z-score normalization: (x - mu) / sigma."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> StandardScalerTransform:
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)


class YeoJohnsonTransform:
    """Yeo-Johnson power transformation to normalize skewed, heavy-tailed distributions."""

    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson', standardize=True)

    def fit(self, X: np.ndarray) -> YeoJohnsonTransform:
        self.pt.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pt.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pt.fit_transform(X)
