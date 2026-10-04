"""Data re-uploading PQC with a controllable number of entangling layers.

Layer l = 0..L-1:  RY(θ_l,q,0) RZ(θ_l,q,1) on every qubit
                   CZ on every topology pair            (only if l < k)
                   RX(λ_l,q · x_q) on qubit q            (data re-upload, trainable scaling)
Final block:       RY RZ on every qubit, then measure.

CZ is parameter-free, so the parameter count is independent of k and topology.
With k = 0 qubit q only ever sees x_q, hence <Z_q> is a function of x_q alone.
"""

from __future__ import annotations

from itertools import combinations

import pennylane as qml
import torch


def entangler_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    if topology == "chain":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    if topology == "all":
        return list(combinations(range(n_qubits), 2))
    raise ValueError(f"unknown topology {topology!r}")


def n_observables(n_qubits: int, obs: str) -> int:
    return n_qubits if obs == "z" else n_qubits + n_qubits * (n_qubits - 1) // 2


def observable_matrix(n_qubits: int, obs: str) -> torch.Tensor:
    """(2^n, d) matrix M with <O_j> = probs @ M[:, j]; PennyLane wire 0 is the most significant bit."""
    idx = torch.arange(2**n_qubits)
    z = torch.stack([1.0 - 2.0 * ((idx >> (n_qubits - 1 - q)) & 1).double() for q in range(n_qubits)], dim=1)
    if obs == "z":
        return z
    if obs == "zzz":
        zz = [z[:, i] * z[:, j] for i, j in combinations(range(n_qubits), 2)]
        return torch.cat([z, torch.stack(zz, dim=1)], dim=1)
    raise ValueError(f"unknown observable set {obs!r}")


def n_params(n_qubits: int, n_layers: int) -> int:
    return 2 * n_qubits * (n_layers + 1) + n_qubits * n_layers


def build_qnode(n_qubits: int, n_layers: int, k: int, topology: str):
    """Returns f(x, theta, lam) -> probs of shape (B, 2^n). x: (B, n); theta: (L+1, n, 2); lam: (L, n)."""
    if not 0 <= k <= n_layers:
        raise ValueError("k out of range")
    pairs = entangler_pairs(n_qubits, topology)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, theta, lam):
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(theta[layer, q, 0], wires=q)
                qml.RZ(theta[layer, q, 1], wires=q)
            if layer < k:
                for a, b in pairs:
                    qml.CZ(wires=[a, b])
            for q in range(n_qubits):
                qml.RX(lam[layer, q] * x[:, q], wires=q)
        for q in range(n_qubits):
            qml.RY(theta[n_layers, q, 0], wires=q)
            qml.RZ(theta[n_layers, q, 1], wires=q)
        return qml.probs(wires=range(n_qubits))

    return circuit
