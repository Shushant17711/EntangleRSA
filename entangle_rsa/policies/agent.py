"""Actor-critic agent assembled from a RunConfig: encoder (quantum | MLP) + head + shared critic."""

from __future__ import annotations

import torch
from torch import nn

from entangle_rsa.config import RunConfig
from entangle_rsa.envs.wrappers import get_task
from entangle_rsa.policies.classical_policy import Critic, MLPEncoder, param_matched_hidden
from entangle_rsa.policies.heads import CategoricalHead, GaussianHead
from entangle_rsa.policies.quantum_policy import QuantumEncoder


def build_encoder(cfg: RunConfig) -> nn.Module:
    n = get_task(cfg.task).obs_dim
    if cfg.family == "quantum":
        return QuantumEncoder(n, cfg.n_layers, cfg.k, cfg.topology, cfg.obs)
    if cfg.family == "mlp":
        return MLPEncoder(n, hidden=64)
    if cfg.family == "mlp-pm":
        return MLPEncoder(n, hidden=param_matched_hidden(n, cfg.n_layers))
    raise ValueError(cfg.family)


class Agent(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        task = get_task(cfg.task)
        self.cfg = cfg
        self.discrete = task.discrete
        self.encoder = build_encoder(cfg)
        head_cls = CategoricalHead if task.discrete else GaussianHead
        self.head = head_cls(self.encoder.out_dim, task.n_actions)
        self.critic = Critic(task.obs_dim)

    def rep(self, x: torch.Tensor) -> torch.Tensor:
        """The representation compared by RSA: pre-head encoder output."""
        return self.encoder(x)

    def dist(self, x: torch.Tensor):
        return self.head(self.rep(x).float())

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Optimizer groups: circuit params get their own LR; classical actor params (MLP encoder, head) share one."""
        enc, head = list(self.encoder.parameters()), list(self.head.parameters())
        groups = {"quantum": enc, "actor": head} if self.cfg.family == "quantum" else {"actor": enc + head}
        groups["critic"] = list(self.critic.parameters())
        return groups

    def encoder_params(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())
