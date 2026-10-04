import csv
import json

import pytest
import torch

from entangle_rsa.config import PPOConfig, RunConfig
from entangle_rsa.rl.ppo import load_agent, train


@pytest.mark.parametrize(
    "task,family,k",
    [("cartpole", "mlp", 0), ("cartpole", "quantum", 2), ("pendulum", "quantum", 1), ("cartpole", "mlp-pm", 0)],
)
def test_two_update_smoke_run_writes_artifacts(tmp_path, task, family, k):
    ppo = PPOConfig(total_steps=2 * 4 * 32, n_envs=4, n_steps=32, n_epochs=1, n_minibatches=2)
    cfg = RunConfig(task=task, family=family, k=k, seed=3, ppo=ppo)
    out = train(cfg, root=tmp_path)
    for f in ("config.json", "metrics.csv", "model.pt", "eval.json", "done"):
        assert (out / f).exists(), f
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert len(rows) == 2
    assert float(rows[-1]["grad_var"]) >= 0
    ev = json.loads((out / "eval.json").read_text())
    assert len(ev["returns"]) == 20
    agent = load_agent(out)
    x = torch.zeros(3, agent.critic.net[0].in_features)
    assert agent.rep(x).shape[0] == 3
