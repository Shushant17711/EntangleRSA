"""E1: validate the RSA pipeline on synthetic problems with known answers (gate G1).

Run directly (`python -m entangle_rsa.rsa.validation_toy_task`) to print a report; the assertions live in
tests/test_validation_e1.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from entangle_rsa.rsa.core import cka_linear, rdm, rsa_from_rdms, rsa_score
from entangle_rsa.rsa.partitioned_analysis import discrepancy, partition_pairs, permutation_test


def target_f(x: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(np.pi * x[:, 0] * x[:, 1]), x[:, 0] ** 2, np.cos(np.pi * x[:, 2]) + x[:, 3]], 1)


def target_g(x: np.ndarray) -> np.ndarray:
    """An unrelated function of the same inputs."""
    return np.stack([np.sin(3 * x[:, 3]) * x[:, 2], np.abs(x[:, 1] - 0.3), np.tanh(4 * x[:, 0])], 1)


class _Net(nn.Module):
    def __init__(self, width: int, depth: int, n_in: int = 4, n_out: int = 3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width, n_out)

    def forward(self, x):
        return self.out(self.body(x))


def train_regressor(fn, width: int, depth: int, seed: int, steps: int = 3000) -> _Net:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(4096, 4)).astype(np.float32)
    y = fn(x).astype(np.float32)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    net = _Net(width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (256,))
        loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def hidden(net: _Net, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net.body(torch.from_numpy(x.astype(np.float32))).numpy()


def same_function_scores(n_probes: int = 300, seed: int = 99, replicate: int = 0) -> dict:
    """Two differently shaped nets trained on f vs a net trained on unrelated g, plus the input-geometry floor."""
    probes = np.random.default_rng(seed + replicate).uniform(-1, 1, size=(n_probes, 4))
    s0 = 10 * replicate
    a = hidden(train_regressor(target_f, width=32, depth=2, seed=s0), probes)
    b = hidden(train_regressor(target_f, width=128, depth=3, seed=s0 + 1), probes)
    c = hidden(train_regressor(target_g, width=64, depth=2, seed=s0 + 2), probes)
    return {
        "rsa_same": rsa_score(a, b),
        "rsa_diff": rsa_score(a, c),
        "rsa_input_floor": rsa_score(a, probes),
        "cka_same": cka_linear(a, b),
        "cka_diff": cka_linear(a, c),
    }


def monotone_warp_invariance(seed: int = 0) -> float:
    x = np.random.default_rng(seed).normal(size=(100, 5))
    d = rdm(x)
    return rsa_from_rdms(d, np.exp(3 * d) - 1)


def noise_curve(levels=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 6))
    return [rsa_score(x, x + s * rng.normal(size=x.shape)) for s in levels]


def planted_coupling(n_probes: int = 250, seed: int = 0, n_perm: int = 500) -> dict:
    """Reference encodes the joint x1*x2; a 'separable' model only has x1, x2 individually.

    Expect the separable model's discrepancy to the reference to concentrate on pairs differing along (x1, x2).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n_probes, 3))
    ref = np.column_stack([3 * x[:, 0] * x[:, 1], x[:, 0], x[:, 1], x[:, 2]])
    sep = np.column_stack([x[:, 0], x[:, 1], x[:, 2]])
    dense = ref + 0.05 * rng.normal(size=ref.shape)
    dense2 = ref + 0.05 * rng.normal(size=ref.shape)
    part = partition_pairs(x, coupled=(0, 1), uncoupled=(2,))
    d_sep, d_dense, d_dense2 = (discrepancy(m, ref, "euclidean") for m in (sep, dense, dense2))
    return {
        "planted": permutation_test(d_sep, d_dense, part, n_perm=n_perm, seed=seed),
        "null": permutation_test(d_dense2, d_dense, part, n_perm=n_perm, seed=seed),
    }


def qubit_permutation_check(seed: int = 0) -> float:
    """Relabelling qubits (and inputs, params) of an all-to-all circuit must leave the RDM unchanged."""
    from entangle_rsa.policies.quantum_policy import QuantumEncoder

    torch.manual_seed(seed)
    n = 4
    a = QuantumEncoder(n, 4, k=4, topology="all")
    b = QuantumEncoder(n, 4, k=4, topology="all")
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        b.theta.copy_(a.theta[:, perm])
        b.lam.copy_(a.lam[:, perm])
        x = torch.rand(80, n) * 2 - 1
        ra, rb = a(x).numpy(), b(x[:, perm]).numpy()
    return rsa_score(ra, rb)


if __name__ == "__main__":
    for r in range(3):
        print("same/different function:", same_function_scores(replicate=r))
    print("monotone warp RSA:", monotone_warp_invariance())
    print("noise curve:", noise_curve())
    print("planted coupling:", planted_coupling())
    print("qubit permutation RSA:", qubit_permutation_check())
