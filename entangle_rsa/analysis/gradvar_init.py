"""Barren-plateau proxy (Req 6.6): variance over random initialisations of ∂C/∂θ, C = mean <Z_0> over probes.

python -m entangle_rsa.analysis.gradvar_init --task cartpole --probes runs/cartpole/probes_trajectory_500_s0.npy
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from entangle_rsa.config import N_LAYERS
from entangle_rsa.envs.wrappers import get_task, scale_obs
from entangle_rsa.policies.quantum_policy import QuantumEncoder

CONDS = [("chain", 0)] + [(t, k) for t in ("chain", "all") for k in range(1, N_LAYERS + 1)]


def init_grad_variance(task_name: str, topology: str, k: int, probes_raw: np.ndarray, n_inits: int = 200, seed: int = 0) -> dict:
    task = get_task(task_name)
    x = torch.from_numpy(scale_obs(task, probes_raw))
    torch.manual_seed(seed)
    enc = QuantumEncoder(task.obs_dim, N_LAYERS, k, topology)
    grads = []
    for _ in range(n_inits):
        with torch.no_grad():
            enc.theta.copy_(torch.rand_like(enc.theta) * 2 * math.pi)
            enc.lam.fill_(1.0)
        enc.zero_grad()
        enc(x)[:, 0].mean().backward()
        grads.append(enc.theta.grad.detach().flatten().clone())
    g = torch.stack(grads).double()
    var = g.var(0)
    return {
        "task": task_name,
        "cond": "q-sep-k0" if k == 0 else f"q-{topology}-k{k}",
        "topology": topology,
        "k": k,
        "grad_var_mean": float(var.mean()),
        "grad_var_first": float(var[0]),
        "n_inits": n_inits,
    }


def run(task_name: str, probes_raw: np.ndarray, out: Path, n_inits: int = 200, n_probe: int = 64) -> list[dict]:
    rng = np.random.default_rng(0)
    batch = probes_raw[rng.choice(len(probes_raw), size=min(n_probe, len(probes_raw)), replace=False)]
    rows = [init_grad_variance(task_name, t, k, batch, n_inits) for t, k in CONDS]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cartpole")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-inits", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out or f"results/gradvar_init_{a.task}.csv")
    for r in run(a.task, np.load(a.probes), out, a.n_inits):
        print(r)
