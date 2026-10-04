"""Policy heads shared by all encoder families: representation -> action distribution."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class CategoricalHead(nn.Module):
    def __init__(self, rep_dim: int, n_actions: int):
        super().__init__()
        self.linear = nn.Linear(rep_dim, n_actions)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, rep: torch.Tensor) -> Categorical:
        return Categorical(logits=self.linear(rep))


class GaussianHead(nn.Module):
    """Diagonal Gaussian with a state-independent log-std; actions are scaled to the env bounds by the trainer."""

    def __init__(self, rep_dim: int, action_dim: int):
        super().__init__()
        self.mean = nn.Linear(rep_dim, action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.5)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, rep: torch.Tensor) -> Normal:
        mean = self.mean(rep)
        return Normal(mean, self.log_std.exp().expand_as(mean))
