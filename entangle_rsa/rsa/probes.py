"""Probe states shared by every model in a comparison (Req 4.2).

Primary: states pooled from rollouts of all trained runs, an equal share per run, so no single policy
defines the probe distribution. Alternative: uniform states in the box spanned by the pooled states.
Probes are stored as RAW observations; models apply the fixed scaling themselves via `scale_obs`.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from entangle_rsa.envs.wrappers import get_task, scale_obs
from entangle_rsa.rl.ppo import env_action, load_agent


def rollout_states(run_dir: Path | str, n_states: int, seed: int, max_episodes: int = 50) -> np.ndarray:
    """Visited raw observations from stochastic rollouts of a trained run."""
    agent = load_agent(run_dir)
    task = get_task(agent.cfg.task)
    env = gym.make(task.gym_id)
    rng = np.random.default_rng(seed)
    states = []
    torch.manual_seed(seed)
    for ep in range(max_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            states.append(np.asarray(obs, dtype=np.float64))
            with torch.no_grad():
                a = agent.dist(torch.from_numpy(scale_obs(task, obs)).unsqueeze(0)).sample().squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(env_action(task, a))
            done = term or trunc
        if len(states) >= 4 * n_states:
            break
    states = np.array(states)
    return states[rng.choice(len(states), size=min(n_states, len(states)), replace=False)]


def trajectory_probes(run_dirs: list[Path], n: int = 500, seed: int = 0) -> np.ndarray:
    run_dirs = sorted(Path(r) for r in run_dirs)
    per_run = int(np.ceil(n / len(run_dirs))) * 2
    pool = np.concatenate([rollout_states(r, per_run, seed + i) for i, r in enumerate(run_dirs)])
    rng = np.random.default_rng(seed)
    return pool[rng.choice(len(pool), size=n, replace=False)]


def uniform_probes(reference: np.ndarray, n: int = 500, seed: int = 0, task_name: str | None = None) -> np.ndarray:
    """Uniform in the 1st–99th percentile box of a reference (pooled) state set."""
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(reference, 1, axis=0), np.percentile(reference, 99, axis=0)
    u = rng.uniform(lo, hi, size=(n, reference.shape[1]))
    # keep (cos, sin) pairs on the unit circle so states are physically valid; angles uniform on [-pi, pi]
    pairs = {"pendulum": [(0, 1)], "acrobot": [(0, 1), (2, 3)]}.get(task_name, [])
    for c, s in pairs:
        ang = rng.uniform(-np.pi, np.pi, size=n)
        u[:, c], u[:, s] = np.cos(ang), np.sin(ang)
    return u


def probe_path(root: Path | str, task: str, kind: str, n: int, seed: int = 0) -> Path:
    return Path(root) / task / f"probes_{kind}_{n}_s{seed}.npy"
