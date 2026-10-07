import numpy as np
import pytest

from entangle_rsa.config import RunConfig
from entangle_rsa.rsa.extract import random_init_reps
from entangle_rsa.rsa.probes import uniform_probes


@pytest.mark.parametrize("family,k,d", [("quantum", 2, 4), ("mlp", 0, 4), ("mlp-pm", 0, 4)])
def test_random_init_reps_shape_and_determinism(family, k, d):
    probes = np.random.default_rng(0).normal(size=(30, 4))
    cfg = RunConfig(family=family, k=k)
    a = random_init_reps(cfg, probes, seed=7)
    b = random_init_reps(cfg, probes, seed=7)
    assert a.shape == (30, d)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, random_init_reps(cfg, probes, seed=8))


def test_uniform_probes_deterministic():
    ref = np.random.default_rng(1).normal(size=(200, 4))
    np.testing.assert_array_equal(uniform_probes(ref, 50, seed=3), uniform_probes(ref, 50, seed=3))
