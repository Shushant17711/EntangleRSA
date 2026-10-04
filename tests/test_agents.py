import pytest
import torch

from entangle_rsa.config import RunConfig
from entangle_rsa.policies.agent import Agent
from entangle_rsa.policies.circuits import n_params
from entangle_rsa.policies.classical_policy import mlp_encoder_params, param_matched_hidden


@pytest.mark.parametrize("task,n", [("cartpole", 4), ("pendulum", 3), ("acrobot", 6)])
def test_param_matched_mlp_within_ten_percent(task, n):
    h = param_matched_hidden(n, 4)
    q = n_params(n, 4)
    assert abs(mlp_encoder_params(n, h) - q) / q <= 0.10


@pytest.mark.parametrize("family", ["quantum", "mlp", "mlp-pm"])
def test_agent_rep_shapes_and_range(family):
    agent = Agent(RunConfig(task="cartpole", family=family, k=2))
    x = torch.randn(8, 4)
    rep = agent.rep(x)
    assert rep.shape == (8, 4)
    assert rep.abs().max() <= 1.0 + 1e-5
    d = agent.dist(x)
    assert d.sample().shape == (8,)
    assert agent.value(x).shape == (8,)


def test_gaussian_agent_on_pendulum():
    agent = Agent(RunConfig(task="pendulum", family="quantum", k=1, topology="all"))
    d = agent.dist(torch.randn(5, 3))
    assert d.sample().shape == (5, 1)


def test_param_groups_partition_all_parameters():
    for family in ("quantum", "mlp"):
        agent = Agent(RunConfig(family=family))
        groups = agent.param_groups()
        ids = [id(p) for ps in groups.values() for p in ps]
        assert len(ids) == len(set(ids)) == len(list(agent.parameters()))
