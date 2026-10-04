import numpy as np
import pytest

from entangle_rsa.rsa.core import rsa_score
from entangle_rsa.rsa.stats import (
    RSAData,
    ci,
    disagreement_verdict,
    hierarchical_bootstrap,
    piecewise_fit,
)


def _family(base, noise, seeds, rng):
    return {s: base + noise * rng.normal(size=base.shape) for s in seeds}


def test_point_estimate_matches_exact_rsa():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(60, 4)), rng.normal(size=(60, 4))
    data = RSAData({"a": {0: a}, "b": {0: b}})
    assert data.score("a", [0], "b", [0]) == pytest.approx(rsa_score(a, b), abs=1e-12)


def test_identical_seeds_give_unit_ceiling():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    data = RSAData({"q": {0: x, 1: x.copy(), 2: x.copy()}, "c": {0: x, 1: x.copy()}})
    s = data.summary("q", "c")
    assert s["rsa_ceil"] == pytest.approx(1.0)
    assert s["rsa_norm"] == pytest.approx(1.0)
    assert s["cka_norm"] == pytest.approx(1.0)


def test_normalisation_corrects_for_seed_noise():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(150, 4))
    data = RSAData({"q": _family(base, 0.8, range(5), rng), "c": _family(base, 0.1, range(5), rng)})
    s = data.summary("q", "c")
    assert s["rsa"] < 0.8 and s["rsa_ceil"] < 0.8
    assert s["rsa_norm"] > s["rsa"]
    assert 0.8 < s["rsa_norm"] < 1.2


def test_bootstrap_ci_covers_truth_and_separates_conditions():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(120, 4))
    other = rng.normal(size=(120, 4))
    data = RSAData(
        {
            "same": _family(base, 0.3, range(4), rng),
            "diff": _family(other, 0.3, range(4), rng),
            "ref": _family(base, 0.3, range(4), rng),
        }
    )
    rows = hierarchical_bootstrap(data, ["same", "diff"], "ref", n_boot=60, seed=0)
    same = ci([r["rsa_norm"] for r in rows if r["cond"] == "same"])
    diff = ci([r["rsa_norm"] for r in rows if r["cond"] == "diff"])
    assert same[0] <= 1.0 + 0.1 and same[1] >= 0.9
    assert diff[1] < same[0]


def test_bootstrap_aggregates_per_seed_metrics():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(40, 3))
    data = RSAData({"q": _family(base, 0.2, range(3), rng), "c": _family(base, 0.2, range(3), rng)})
    rows = hierarchical_bootstrap(data, ["q"], "c", per_seed={"q": {"ret": [1.0, 1.0, 1.0]}}, n_boot=5)
    assert all(r["ret"] == 1.0 for r in rows)


def test_piecewise_fit_recovers_planted_breakpoint():
    x = np.arange(0, 5.0)
    y = np.minimum(x, 2.0) * 0.3 + 0.1
    fit = piecewise_fit(x, y)
    assert fit["breakpoint"] == pytest.approx(2.0, abs=0.15)
    assert fit["slope1"] == pytest.approx(0.3, abs=1e-6)
    assert abs(fit["slope2"]) < 1e-6


def test_disagreement_rule():
    assert disagreement_verdict((0.2, 0.9), (-0.9, -0.3)) == "metric-dependent"
    assert disagreement_verdict((0.2, 0.9), (-0.9, 0.3)) == "consistent"
    assert disagreement_verdict((0.2, 0.9), (0.1, 0.3)) == "consistent"


def test_bootstrap_independent_of_worker_count():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(40, 3))
    data = RSAData({"q": _family(base, 0.3, range(3), rng), "c": _family(base, 0.3, range(3), rng)})
    a = hierarchical_bootstrap(data, ["q"], "c", n_boot=6, workers=1)
    b = hierarchical_bootstrap(data, ["q"], "c", n_boot=6, workers=3)
    assert [r["rsa_norm"] for r in a] == pytest.approx([r["rsa_norm"] for r in b], nan_ok=True)
