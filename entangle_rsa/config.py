"""Run configuration: the single contract shared by training, sweeps and analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNS_ROOT = Path("runs")

FAMILIES = ("quantum", "mlp", "mlp-pm")
TOPOLOGIES = ("chain", "all")
OBSERVABLES = ("z", "zzz")
N_LAYERS = 4


@dataclass
class PPOConfig:
    total_steps: int = 500_000
    n_envs: int = 8
    n_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    n_minibatches: int = 4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_actor: float = 5e-4  # classical actor params (MLP layers, policy head)
    lr_quantum: float = 5e-3  # circuit angles and input scalings
    lr_critic: float = 2.5e-3
    anneal_lr: bool = True


@dataclass
class RunConfig:
    task: str = "cartpole"
    family: str = "quantum"
    topology: str = "chain"
    k: int = 0  # number of entangling layers (quantum only)
    n_layers: int = N_LAYERS
    obs: str = "z"  # observable set (quantum only)
    seed: int = 0
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        if self.obs not in OBSERVABLES:
            raise ValueError(f"obs must be one of {OBSERVABLES}")
        if not 0 <= self.k <= self.n_layers:
            raise ValueError(f"k must be in [0, {self.n_layers}]")
        if self.family == "quantum" and self.k == 0:
            self.topology = "chain"  # canonical: separable circuits have no topology

    @property
    def cond_name(self) -> str:
        if self.family != "quantum":
            return self.family
        topo = "sep" if self.k == 0 else self.topology
        return f"q-{topo}-k{self.k}"

    @property
    def variant(self) -> str:
        """Condition plus observable set; the directory name of a run group."""
        return f"{self.cond_name}-{self.obs}" if self.family == "quantum" else self.cond_name

    def run_dir(self, root: Path | str = RUNS_ROOT) -> Path:
        return Path(root) / self.task / self.variant / f"seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        ppo = PPOConfig(**d.pop("ppo", {}))
        return cls(ppo=ppo, **d)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))


def task_defaults(task: str) -> PPOConfig:
    """Per-task PPO defaults, frozen after the pilot (never tuned per condition)."""
    if task == "pendulum":
        return PPOConfig(total_steps=500_000, n_envs=16, n_steps=64, gamma=0.9, ent_coef=0.0, n_epochs=10)
    return PPOConfig()
