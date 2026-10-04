"""Representation extraction over a fixed probe set (Req 4.3, 4.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from entangle_rsa.config import RunConfig
from entangle_rsa.envs.wrappers import get_task, scale_obs
from entangle_rsa.policies.agent import Agent
from entangle_rsa.rl.ppo import load_agent


def agent_reps(agent: Agent, probes_raw: np.ndarray) -> np.ndarray:
    task = get_task(agent.cfg.task)
    x = torch.from_numpy(scale_obs(task, probes_raw))
    with torch.no_grad():
        return agent.rep(x).double().numpy()


def run_reps(run_dir: Path | str, probes_raw: np.ndarray) -> np.ndarray:
    """Quantum: measured expectation values; MLP: penultimate (tanh) activations."""
    return agent_reps(load_agent(run_dir), probes_raw)


def random_init_reps(cfg: RunConfig, probes_raw: np.ndarray, seed: int) -> np.ndarray:
    """Untrained agent with the same architecture — the capacity/architecture baseline."""
    torch.manual_seed(seed)
    return agent_reps(Agent(cfg), probes_raw)
