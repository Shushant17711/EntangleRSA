import numpy as np
import pytest

from entangle_rsa.config import RunConfig
from entangle_rsa.envs.wrappers import get_task, make_env, physical_features, scale_obs


def test_separable_condition_is_shared_across_topologies():
    a = RunConfig(family="quantum", topology="chain", k=0)
    b = RunConfig(family="quantum", topology="all", k=0)
    assert a.cond_name == b.cond_name == "q-sep-k0"
    assert a.run_dir() == b.run_dir()


def test_condition_names():
    assert RunConfig(family="quantum", topology="all", k=3).cond_name == "q-all-k3"
    assert RunConfig(family="quantum", topology="chain", k=2, obs="zzz").variant == "q-chain-k2-zzz"
    assert RunConfig(family="mlp").variant == "mlp"
    assert str(RunConfig(family="mlp-pm", seed=4).run_dir()) == "runs/cartpole/mlp-pm/seed4"


def test_config_json_roundtrip():
    c = RunConfig(task="pendulum", family="quantum", topology="all", k=2, seed=7)
    c.ppo.total_steps = 1234
    assert RunConfig.from_json(c.to_json()) == c


def test_invalid_config():
    with pytest.raises(ValueError):
        RunConfig(k=5)
    with pytest.raises(ValueError):
        RunConfig(family="transformer")


def test_scaled_cartpole_obs_is_order_one():
    task = get_task("cartpole")
    env = make_env("cartpole", seed=0)
    obs, _ = env.reset(seed=0)
    xs = []
    for _ in range(500):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        xs.append(scale_obs(task, obs))
        if term or trunc:
            obs, _ = env.reset()
    xs = np.array(xs)
    assert np.abs(xs).max() < 3.0


def test_pendulum_physical_features():
    task = get_task("pendulum")
    obs = np.array([[np.cos(0.5), np.sin(0.5), 2.0]])
    np.testing.assert_allclose(physical_features(task, obs), [[0.5, 2.0]])
