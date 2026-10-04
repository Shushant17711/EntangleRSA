"""Independent pure-torch statevector simulator of the circuit in `circuits.py`.

Used only as a test oracle for the PennyLane implementation: it shares no simulation code with it.
"""

from __future__ import annotations

import torch

from entangle_rsa.policies.circuits import entangler_pairs, observable_matrix

CDTYPE = torch.complex128


def _rot(kind: str, t: torch.Tensor) -> torch.Tensor:
    """Batched single-qubit rotation matrices, t: (B,) -> (B, 2, 2)."""
    t = t.to(torch.float64)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    zero = torch.zeros_like(t)
    if kind == "RX":
        re = torch.stack([torch.stack([c, zero], -1), torch.stack([zero, c], -1)], -2)
        im = torch.stack([torch.stack([zero, -s], -1), torch.stack([-s, zero], -1)], -2)
    elif kind == "RY":
        re = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        im = torch.zeros_like(re)
    elif kind == "RZ":
        re = torch.stack([torch.stack([c, zero], -1), torch.stack([zero, c], -1)], -2)
        im = torch.stack([torch.stack([-s, zero], -1), torch.stack([zero, s], -1)], -2)
    else:
        raise ValueError(kind)
    return torch.complex(re, im)


def _apply_1q(psi: torch.Tensor, u: torch.Tensor, q: int) -> torch.Tensor:
    psi = psi.movedim(q + 1, -1)
    psi = torch.einsum("b...j,bij->b...i", psi, u)
    return psi.movedim(-1, q + 1)


def _cz_phase(n: int, pairs) -> torch.Tensor:
    idx = torch.arange(2**n)
    bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
    sign = torch.ones(2**n, dtype=torch.float64)
    for a, b in pairs:
        sign = sign * torch.where((bits[a] & bits[b]) == 1, -1.0, 1.0).double()
    return sign.to(CDTYPE).reshape([2] * n)


def simulate_probs(x, theta, lam, n_layers: int, k: int, topology: str) -> torch.Tensor:
    batch, n = x.shape
    psi = torch.zeros((batch,) + (2,) * n, dtype=CDTYPE)
    psi[(slice(None),) + (0,) * n] = 1.0
    phase = _cz_phase(n, entangler_pairs(n, topology))
    ones = torch.ones(batch, dtype=torch.float64)

    def var_block(layer):
        nonlocal psi
        for q in range(n):
            psi = _apply_1q(psi, _rot("RY", theta[layer, q, 0] * ones), q)
            psi = _apply_1q(psi, _rot("RZ", theta[layer, q, 1] * ones), q)

    for layer in range(n_layers):
        var_block(layer)
        if layer < k:
            psi = psi * phase
        for q in range(n):
            psi = _apply_1q(psi, _rot("RX", lam[layer, q] * x[:, q]), q)
    var_block(n_layers)
    return (psi.abs() ** 2).reshape(batch, -1)


def simulate_expvals(x, theta, lam, n_layers: int, k: int, topology: str, obs: str = "z") -> torch.Tensor:
    probs = simulate_probs(x, theta, lam, n_layers, k, topology)
    return probs @ observable_matrix(x.shape[1], obs).to(probs.dtype)
