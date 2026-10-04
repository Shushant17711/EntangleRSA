"""Core representational-similarity primitives: RDMs, Spearman RSA, linear CKA.

All functions take representations as arrays of shape (n_probes, n_features). Two systems are
compared RDM-to-RDM, so their feature dimensionalities may differ.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

METRICS = ("cosine", "correlation", "euclidean")


def rdm(x: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Representational dissimilarity matrix, shape (n, n), zero diagonal."""
    x = np.asarray(x, dtype=np.float64)
    if metric == "euclidean":
        sq = (x**2).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
        d = np.sqrt(d2)
    elif metric in ("cosine", "correlation"):
        if metric == "correlation":
            x = x - x.mean(1, keepdims=True)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        # Constant (zero-norm) rows get distance 1 to everything: no direction information.
        xn = np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12)
        d = 1.0 - xn @ xn.T
        zero = norm[:, 0] <= 1e-12
        d[zero, :] = 1.0
        d[:, zero] = 1.0
    else:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    np.fill_diagonal(d, 0.0)
    return d


def upper_tri(m: np.ndarray) -> np.ndarray:
    """Strict upper triangle of a square matrix, row-major."""
    i, j = np.triu_indices(m.shape[0], k=1)
    return m[i, j]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def rsa_from_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    return spearman(upper_tri(rdm_a), upper_tri(rdm_b))


def rsa_score(x: np.ndarray, y: np.ndarray, metric: str = "correlation") -> float:
    """Spearman correlation between the upper triangles of the two systems' RDMs."""
    return rsa_from_rdms(rdm(x, metric), rdm(y, metric))


def cka_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centred kernel alignment (Kornblith et al. 2019)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0)
    y = y - y.mean(0)
    num = np.linalg.norm(y.T @ x, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den) if den > 0 else 0.0
