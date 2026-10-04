import numpy as np
import pytest
from scipy.stats import spearmanr

from entangle_rsa.rsa.core import cka_linear, rdm, rsa_score, upper_tri


def test_rdm_cosine_matches_manual():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 5))
    d = rdm(x, metric="cosine")
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    np.testing.assert_allclose(d, 1 - xn @ xn.T, atol=1e-10)
    assert d.shape == (10, 10)
    np.testing.assert_allclose(np.diag(d), 0, atol=1e-10)


def test_rdm_correlation_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 6))
    np.testing.assert_allclose(rdm(x, metric="correlation"), 1 - np.corrcoef(x), atol=1e-10)


def test_rdm_euclidean():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(rdm(x, metric="euclidean")[0, 1], 5.0)


def test_upper_tri_excludes_diagonal():
    m = np.arange(16).reshape(4, 4)
    assert list(upper_tri(m)) == [1, 2, 3, 6, 7, 11]


def test_rsa_identical_and_rotation_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert rsa_score(x, x) == pytest.approx(1.0)
    # Euclidean RDMs are invariant to rotation
    assert rsa_score(x, x @ q, metric="euclidean") == pytest.approx(1.0)


def test_rsa_handles_different_dimensionality():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 3))
    y = np.concatenate([x, np.zeros((40, 5))], axis=1)  # same geometry, padded
    assert rsa_score(x, y, metric="euclidean") == pytest.approx(1.0)


def test_rsa_matches_scipy_spearman():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 7))
    expected = spearmanr(upper_tri(rdm(x)), upper_tri(rdm(y))).statistic
    assert rsa_score(x, y) == pytest.approx(expected)


def test_rsa_independent_is_near_zero():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(200, 4)), rng.normal(size=(200, 4))
    assert abs(rsa_score(x, y)) < 0.1


def test_cka_properties():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(100, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert cka_linear(x, x) == pytest.approx(1.0)
    assert cka_linear(x, 3.0 * x @ q) == pytest.approx(1.0)  # rotation + isotropic scale
    assert cka_linear(x, rng.normal(size=(100, 8))) < 0.2
