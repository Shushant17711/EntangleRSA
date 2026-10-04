import math

import pytest
import torch

from entangle_rsa.policies.circuits import build_qnode, n_params, observable_matrix
from entangle_rsa.policies.quantum_policy import QuantumEncoder
from entangle_rsa.policies.statevector import simulate_expvals

N, L = 4, 4


def _random_params(gen):
    theta = (torch.rand(L + 1, N, 2, generator=gen, dtype=torch.float64) * 2 * math.pi).requires_grad_()
    lam = (torch.rand(L, N, generator=gen, dtype=torch.float64) * 2).requires_grad_()
    x = torch.randn(6, N, generator=gen, dtype=torch.float64)
    return x, theta, lam


@pytest.mark.parametrize("topology", ["chain", "all"])
@pytest.mark.parametrize("obs", ["z", "zzz"])
def test_pennylane_matches_statevector_oracle(topology, obs):
    gen = torch.Generator().manual_seed(123)
    m = observable_matrix(N, obs)
    for trial in range(20):
        k = trial % (L + 1)
        x, theta, lam = _random_params(gen)
        qnode = build_qnode(N, L, k, topology)
        pl = qnode(x, theta, lam) @ m
        ref = simulate_expvals(x, theta, lam, L, k, topology, obs)
        assert torch.allclose(pl, ref, atol=1e-5), (k, (pl - ref).abs().max())
        w = torch.randn(pl.shape, generator=gen, dtype=torch.float64)
        g_pl = torch.autograd.grad((pl * w).sum(), (theta, lam))
        g_ref = torch.autograd.grad((ref * w).sum(), (theta, lam))
        for a, b in zip(g_pl, g_ref):
            assert torch.allclose(a, b, atol=1e-4)


def test_parameter_count_independent_of_entanglement():
    counts = {
        sum(p.numel() for p in QuantumEncoder(N, L, k, topo).parameters())
        for k in range(L + 1)
        for topo in ("chain", "all")
    }
    assert counts == {n_params(N, L)} == {56}


def _input_jacobian(enc, x):
    return torch.autograd.functional.jacobian(lambda v: enc(v.unsqueeze(0)).squeeze(0), x)


def test_separable_circuit_z_depends_only_on_own_input():
    torch.manual_seed(0)
    enc = QuantumEncoder(N, L, k=0, topology="chain")
    for _ in range(5):
        jac = _input_jacobian(enc, torch.randn(N))  # (out=N, in=N)
        off = jac - torch.diag(torch.diagonal(jac))
        assert off.abs().max() < 1e-6


def test_entangled_circuit_mixes_inputs():
    torch.manual_seed(0)
    enc = QuantumEncoder(N, L, k=L, topology="all")
    jac = _input_jacobian(enc, torch.randn(N))
    off = jac - torch.diag(torch.diagonal(jac))
    assert off.abs().max() > 1e-3


def test_encoder_shapes_and_range():
    enc = QuantumEncoder(N, L, k=2, topology="chain", obs="zzz")
    out = enc(torch.randn(16, N))
    assert out.shape == (16, 10)
    assert out.abs().max() <= 1 + 1e-5
    assert enc(torch.randn(1, N)).shape == (1, 10)
    assert enc(torch.randn(N)).shape == (10,)
