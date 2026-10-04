"""PQC actor: scaled observation -> expectation-value representation -> policy head."""

from __future__ import annotations

import math

import torch
from torch import nn

from entangle_rsa.policies.circuits import build_qnode, n_observables, observable_matrix


class QuantumEncoder(nn.Module):
    """Maps x (B, n) to the measured expectation values (B, d) — the representation used by RSA."""

    def __init__(self, n_qubits: int, n_layers: int, k: int, topology: str, obs: str = "z"):
        super().__init__()
        self.n_qubits, self.n_layers, self.k, self.topology, self.obs = n_qubits, n_layers, k, topology, obs
        self.out_dim = n_observables(n_qubits, obs)
        self.theta = nn.Parameter(torch.rand(n_layers + 1, n_qubits, 2) * 2 * math.pi)
        self.lam = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.register_buffer("obs_matrix", observable_matrix(n_qubits, obs).float(), persistent=False)
        self._qnode = build_qnode(n_qubits, n_layers, k, topology)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        probs = self._qnode(x, self.theta, self.lam)
        if probs.dim() == 1:  # PennyLane drops the broadcast dim for batch size 1 in some versions
            probs = probs.unsqueeze(0)
        rep = probs.to(self.obs_matrix.dtype) @ self.obs_matrix
        return rep.squeeze(0) if squeeze else rep
