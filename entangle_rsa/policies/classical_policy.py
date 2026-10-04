"""Classical encoders and the shared critic.

Matching criteria (stated explicitly, see README):
- primary `mlp`: representation width = n_qubits and bounded range (tanh), like the quantum <Z_i> vector;
  hidden width 64, so it has more parameters than the circuit.
- secondary `mlp-pm`: same shape, hidden width chosen so the encoder parameter count matches the circuit's
  (within ±10%). Policy heads and critics are identical across families, so only encoders differ.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from entangle_rsa.policies.circuits import n_params as quantum_n_params


def _layer(i: int, o: int, gain: float = np.sqrt(2)) -> nn.Linear:
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)
    return lin


def mlp_encoder_params(n_in: int, hidden: int) -> int:
    return n_in * hidden + hidden + hidden * n_in + n_in


def param_matched_hidden(n_qubits: int, n_layers: int) -> int:
    target = quantum_n_params(n_qubits, n_layers)
    return min(range(1, 257), key=lambda h: abs(mlp_encoder_params(n_qubits, h) - target))


class MLPEncoder(nn.Module):
    """x (B, n) -> tanh(W2 tanh(W1 x)) of width n — the penultimate layer used as the representation."""

    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.out_dim = n_in
        self.net = nn.Sequential(_layer(n_in, hidden), nn.Tanh(), _layer(hidden, n_in), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _layer(n_in, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, gain=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
