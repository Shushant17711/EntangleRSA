"""H3: is the quantum-vs-classical representational gap localised to the coupled degrees of freedom?

For every probe pair (i, j) we compute
  * a discrepancy D_ij = |r(RDM_model)_ij - r(RDM_ref)_ij|, r = rank of the upper triangle scaled to [0, 1];
  * which physical subspace the pair differs along (coupled vs uncoupled), from standardised feature gaps.
Pairs are binned by overall gap magnitude (deciles) so coupled and uncoupled pairs are compared at matched scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from entangle_rsa.rsa.core import rdm, upper_tri


@dataclass
class PairPartition:
    labels: np.ndarray  # per upper-triangle pair: 1 coupled, 0 uncoupled, -1 neither
    bins: np.ndarray  # magnitude decile per pair


def partition_pairs(
    feats: np.ndarray,
    coupled: tuple[int, ...],
    uncoupled: tuple[int, ...],
    hi: float = 0.75,
    lo: float = 0.25,
    n_bins: int = 10,
) -> PairPartition:
    f = np.asarray(feats, dtype=np.float64)
    f = (f - f.mean(0)) / np.where(f.std(0) > 1e-12, f.std(0), 1.0)
    i, j = np.triu_indices(len(f), k=1)
    d2 = (f[i] - f[j]) ** 2
    if uncoupled:
        c = d2[:, list(coupled)].sum(1)
        u = d2[:, list(uncoupled)].sum(1)
        share = c / np.maximum(c + u, 1e-12)
        mag = c + u
    else:
        # Only one physical pair (e.g. Pendulum): "coupled" pairs differ jointly in both dims,
        # "uncoupled" pairs differ along essentially one of them.
        a, b = d2[:, coupled[0]], d2[:, coupled[1]]
        share = 2 * np.minimum(a, b) / np.maximum(a + b, 1e-12)
        mag = a + b
        hi, lo = 0.5, 0.1
    labels = np.full(len(share), -1)
    labels[share >= hi] = 1
    labels[share <= lo] = 0
    edges = np.quantile(mag, np.linspace(0, 1, n_bins + 1)[1:-1])
    return PairPartition(labels=labels, bins=np.searchsorted(edges, mag))


def rank01(v: np.ndarray) -> np.ndarray:
    return (rankdata(v) - 1) / max(len(v) - 1, 1)


def discrepancy(rep_model: np.ndarray, rep_ref: np.ndarray, metric: str = "correlation") -> np.ndarray:
    return np.abs(rank01(upper_tri(rdm(rep_model, metric))) - rank01(upper_tri(rdm(rep_ref, metric))))


def matched_gap(d: np.ndarray, part: PairPartition) -> float:
    """Mean over magnitude bins of [mean D on coupled pairs - mean D on uncoupled pairs]."""
    diffs = []
    for b in np.unique(part.bins):
        m = part.bins == b
        c, u = d[m & (part.labels == 1)], d[m & (part.labels == 0)]
        if len(c) and len(u):
            diffs.append(c.mean() - u.mean())
    return float(np.mean(diffs)) if diffs else float("nan")


def permutation_test(
    d_a: np.ndarray, d_b: np.ndarray, part: PairPartition, n_perm: int = 10_000, seed: int = 0
) -> dict:
    """T = gap(D_a - D_b); null shuffles coupled/uncoupled labels within magnitude bins. One-sided (T > 0)."""
    delta = d_a - d_b
    t_obs = matched_gap(delta, part)
    rng = np.random.default_rng(seed)
    sel = part.labels >= 0
    idx_by_bin = [np.flatnonzero(sel & (part.bins == b)) for b in np.unique(part.bins)]
    null = np.empty(n_perm)
    labels = part.labels.copy()
    for p in range(n_perm):
        for idx in idx_by_bin:
            labels[idx] = rng.permutation(part.labels[idx])
        null[p] = matched_gap(delta, PairPartition(labels, part.bins))
    pval = (1 + np.sum(null >= t_obs)) / (1 + n_perm)
    return {"T": t_obs, "p": float(pval), "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null))}
