"""Representation transforms used by the invariance diagnostics (design §8, Req 4.5)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > 1e-12, sd, 1.0)


def pca_whiten(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    keep = s > eps * max(s.max(), 1e-300)
    return u[:, keep] * np.sqrt(len(x) - 1)


def rank_transform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack([rankdata(col) / len(col) for col in x.T])


TRANSFORMS = {"raw": lambda x: np.asarray(x, dtype=np.float64), "zscore": zscore, "whiten": pca_whiten, "rank": rank_transform}
