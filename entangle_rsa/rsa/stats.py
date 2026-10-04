"""Noise ceilings, normalised RSA/CKA, hierarchical bootstrap, saturation fits (Req 4.4, 7.1–7.3, 7.5).

Bootstrap efficiency: each model's RDM upper triangle is ranked once on the full probe set. A probe
resample re-indexes those pairs (dropping pairs of a probe with its own duplicate) and correlates the
pre-computed ranks (Pearson on full-set ranks ≈ Spearman on the resample; exact at the point estimate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, spearmanr

from entangle_rsa.rsa.core import cka_linear, rdm, upper_tri


def _condensed_index(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Index into the row-major strict upper triangle for pairs a < b."""
    return a * n - a * (a + 1) // 2 + (b - a - 1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else 0.0


@dataclass
class ModelRep:
    cond: str
    seed: int
    rep: np.ndarray
    ranks: np.ndarray  # ranks of the RDM upper triangle on the full probe set


class RSAData:
    """Representations for many (condition, seed) models over one shared probe set."""

    def __init__(self, reps: dict[str, dict[int, np.ndarray]], metric: str = "correlation"):
        self.metric = metric
        self.models: dict[str, list[ModelRep]] = {}
        n = None
        for cond, by_seed in reps.items():
            self.models[cond] = []
            for seed, r in sorted(by_seed.items()):
                n = len(r) if n is None else n
                assert len(r) == n, "all models must share the probe set"
                ranks = rankdata(upper_tri(rdm(r, metric))).astype(np.float64)
                self.models[cond].append(ModelRep(cond, seed, np.asarray(r, np.float64), ranks))
        self.n_probes = n

    def pair_index(self, probe_idx: np.ndarray | None) -> np.ndarray | None:
        if probe_idx is None:
            return None
        i, j = np.triu_indices(len(probe_idx), k=1)
        a, b = probe_idx[i], probe_idx[j]
        keep = a != b
        a, b = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
        return _condensed_index(a, b, self.n_probes)

    def _rsa(self, ma: ModelRep, mb: ModelRep, lin) -> float:
        if lin is None:
            return _pearson(ma.ranks, mb.ranks)  # exact Spearman (ranks of the full set)
        return _pearson(ma.ranks[lin], mb.ranks[lin])

    @staticmethod
    def _cka(ma: ModelRep, mb: ModelRep, probe_idx) -> float:
        if probe_idx is None:
            return cka_linear(ma.rep, mb.rep)
        return cka_linear(ma.rep[probe_idx], mb.rep[probe_idx])

    def score(self, ca, sa, cb, sb, probe_idx=None, lin=None, measure="rsa") -> float:
        """Mean similarity over model pairs. Within a condition only unordered pairs of distinct seeds count."""
        if measure == "rsa" and lin is None and probe_idx is not None:
            lin = self.pair_index(probe_idx)
        sa, sb = list(sa), list(sb)
        if ca == cb:
            pairs = [(sa[i], sa[j]) for i in range(len(sa)) for j in range(i + 1, len(sa))]
        else:
            pairs = [(ia, ib) for ia in sa for ib in sb]
        vals = []
        for ia, ib in pairs:
            ma, mb = self.models[ca][ia], self.models[cb][ib]
            if ca == cb and ma.seed == mb.seed:
                continue  # a resampled seed paired with its own duplicate
            vals.append(self._rsa(ma, mb, lin) if measure == "rsa" else self._cka(ma, mb, probe_idx))
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self, cond, ref, seeds_c=None, seeds_r=None, probe_idx=None, ref_ceils: dict | None = None) -> dict:
        sc = range(len(self.models[cond])) if seeds_c is None else seeds_c
        sr = range(len(self.models[ref])) if seeds_r is None else seeds_r
        lin = self.pair_index(probe_idx)
        out = {}
        for m in ("rsa", "cka"):
            cross = self.score(cond, sc, ref, sr, probe_idx, lin, m)
            ceil_c = self.score(cond, sc, cond, sc, probe_idx, lin, m)
            ceil_r = ref_ceils[m] if ref_ceils else self.score(ref, sr, ref, sr, probe_idx, lin, m)
            prod = ceil_c * ceil_r
            out[m] = cross
            out[f"{m}_ceil"] = ceil_c
            out[f"{m}_ref_ceil"] = ceil_r
            out[f"{m}_norm"] = cross / np.sqrt(prod) if prod > 0 else float("nan")
        return out


def _boot_chunk(args):
    data, conds, ref, per_seed, boots, seed = args
    rows = []
    for b in boots:
        rng = np.random.default_rng([seed, b])
        probe_idx = rng.integers(0, data.n_probes, size=data.n_probes)
        seeds = {c: rng.integers(0, len(data.models[c]), size=len(data.models[c])) for c in sorted(set(conds) | {ref})}
        lin = data.pair_index(probe_idx)
        ref_ceils = {m: data.score(ref, seeds[ref], ref, seeds[ref], probe_idx, lin, m) for m in ("rsa", "cka")}
        for c in conds:
            row = {"boot": b, "cond": c}
            row.update(data.summary(c, ref, seeds[c], seeds[ref], probe_idx, ref_ceils))
            if per_seed and c in per_seed:
                for name, vals in per_seed[c].items():
                    row[name] = float(np.mean(np.asarray(vals)[seeds[c]]))
            rows.append(row)
    return rows


def hierarchical_bootstrap(
    data: RSAData,
    conds: list[str],
    ref: str,
    per_seed: dict[str, dict[str, list[float]]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    workers: int = 1,
) -> list[dict]:
    """Resample seeds (per condition), then probes; recompute ceilings and scores in every resample.

    `per_seed[cond][metric]` lists scalar per-seed metrics (e.g. eval return) aligned with data.models[cond];
    they are averaged over the same resampled seeds. Resample b uses RNG seed (seed, b), so results do not
    depend on `workers`.
    """
    chunks = [list(range(n_boot))[i::workers] for i in range(workers)]
    args = [(data, conds, ref, per_seed, ch, seed) for ch in chunks if ch]
    if workers == 1:
        rows = _boot_chunk(args[0])
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            rows = [r for part in pool.map(_boot_chunk, args) for r in part]
    return sorted(rows, key=lambda r: (r["boot"], conds.index(r["cond"])))


def piecewise_fit(x: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Two-segment continuous linear fit y = a + b1*min(x,c) + b2*max(x-c,0); least squares over a breakpoint grid."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if grid is None:
        grid = np.arange(x.min() + 0.5, x.max() - 0.5 + 1e-9, 0.1)
    best = None
    for c in grid:
        X = np.column_stack([np.ones_like(x), np.minimum(x, c), np.maximum(x - c, 0)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"breakpoint": float(c), "a": coef[0], "slope1": coef[1], "slope2": coef[2], "sse": sse}
    return best


def trend(x, y) -> float:
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def ci(vals, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def excludes_zero(interval) -> bool:
    lo, hi = interval
    return bool(lo > 0 or hi < 0)


def disagreement_verdict(rsa_trend_ci, cka_trend_ci) -> str:
    """Pre-registered rule (Req 7.5)."""
    if excludes_zero(rsa_trend_ci) and excludes_zero(cka_trend_ci) and np.sign(rsa_trend_ci[0]) != np.sign(cka_trend_ci[0]):
        return "metric-dependent"
    return "consistent"
