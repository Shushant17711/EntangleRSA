import numpy as np

from entangle_rsa.analysis.gradvar_init import init_grad_variance
from entangle_rsa.rsa.probes import uniform_probes


def test_init_grad_variance_positive_and_finite():
    probes = np.random.default_rng(0).normal(size=(16, 4)) * [1, 1, 0.1, 1]
    r = init_grad_variance("cartpole", "all", 2, probes, n_inits=20)
    assert np.isfinite(r["grad_var_mean"]) and r["grad_var_mean"] > 0
    assert r["cond"] == "q-all-k2"


def test_uniform_probes_respect_box_and_unit_circle():
    ref = np.random.default_rng(1).normal(size=(500, 3))
    ref[:, :2] /= np.linalg.norm(ref[:, :2], axis=1, keepdims=True)
    u = uniform_probes(ref, n=200, seed=0, task_name="pendulum")
    assert u.shape == (200, 3)
    np.testing.assert_allclose(np.hypot(u[:, 0], u[:, 1]), 1.0)
    assert u[:, 2].min() >= np.percentile(ref[:, 2], 1) - 1e-9
