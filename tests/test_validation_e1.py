"""Gate G1: the RSA pipeline must recover known answers before any real experiment (design §9)."""

import numpy as np
import pytest

from entangle_rsa.rsa import validation_toy_task as e1
from entangle_rsa.rsa.core import cka_linear, rsa_score
from entangle_rsa.rsa.transforms import TRANSFORMS, pca_whiten, rank_transform, zscore


@pytest.mark.slow
def test_same_function_networks_score_higher_than_unrelated():
    # Amended criterion (see design §9): unrelated nets share input geometry, so RSA has a floor ~0.4;
    # the test is the replicated gap, not an absolute low value.
    runs = [e1.same_function_scores(replicate=r) for r in range(3)]
    same = np.mean([r["rsa_same"] for r in runs])
    diff = np.mean([r["rsa_diff"] for r in runs])
    assert same >= 0.6
    assert same - diff >= 0.2
    assert all(r["rsa_same"] > r["rsa_diff"] for r in runs)


def test_rsa_invariant_to_monotone_warp():
    assert e1.monotone_warp_invariance() == pytest.approx(1.0)


def test_cka_invariant_to_rotation_and_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert cka_linear(x, 2.5 * x @ q) == pytest.approx(1.0)


def test_rsa_decreases_with_noise():
    curve = e1.noise_curve()
    assert curve[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(curve, curve[1:]))


def test_planted_coupling_is_detected_and_null_is_not():
    out = e1.planted_coupling()
    assert out["planted"]["T"] > 0 and out["planted"]["p"] < 0.01
    assert out["null"]["p"] > 0.05


def test_qubit_relabelling_preserves_geometry():
    assert e1.qubit_permutation_check() == pytest.approx(1.0, abs=1e-6)


def test_transforms():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4)) * [1, 10, 100, 0.1]
    z = zscore(x)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-9)
    w = pca_whiten(x)
    np.testing.assert_allclose(np.cov(w.T), np.eye(4), atol=1e-8)
    r = rank_transform(x)
    assert r.min() > 0 and r.max() == 1
    # rank-transform makes RSA invariant to monotone per-unit warps
    assert rsa_score(rank_transform(np.exp(x)), r) == pytest.approx(1.0)
    assert set(TRANSFORMS) == {"raw", "zscore", "whiten", "rank"}
