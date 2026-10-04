"""Collect finished runs, build/load probes, and extract representations for one task."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from entangle_rsa.config import RunConfig
from entangle_rsa.rsa.extract import random_init_reps, run_reps
from entangle_rsa.rsa.probes import probe_path, trajectory_probes, uniform_probes


def finished_runs(root: Path | str, task: str, obs: str = "z") -> pd.DataFrame:
    rows = []
    for done in sorted(Path(root, task).glob("*/seed*/done")):
        d = done.parent
        cfg = RunConfig.from_json((d / "config.json").read_text())
        if cfg.family == "quantum" and cfg.obs != obs:
            continue
        ev = json.loads((d / "eval.json").read_text())
        m = pd.read_csv(d / "metrics.csv")
        tail = m.tail(max(1, len(m) // 10))
        rows.append(
            {
                "run_dir": d.as_posix(),
                "cond": cfg.cond_name,
                "family": cfg.family,
                "topology": cfg.topology if cfg.k > 0 else "sep",
                "k": cfg.k if cfg.family == "quantum" else np.nan,
                "seed": cfg.seed,
                "eval_return": ev["eval_return_mean"],
                "train_return_final": float(tail["ep_return_mean"].mean()),
                "train_grad_var": float(m["grad_var"].mean()),
                "train_seconds": ev.get("train_seconds", np.nan),
            }
        )
    return pd.DataFrame(rows)


def get_probes(root: Path | str, task: str, runs: pd.DataFrame, kind: str = "trajectory", n: int = 500, seed: int = 0) -> np.ndarray:
    path = probe_path(root, task, kind, n, seed)
    if path.exists():
        return np.load(path)
    if kind == "trajectory":
        # pool from every run of the primary (z-observable) grid, equal share per run
        probes = trajectory_probes([Path(r) for r in runs["run_dir"]], n=n, seed=seed)
    elif kind == "uniform":
        ref = get_probes(root, task, runs, "trajectory", 1000, seed)
        probes = uniform_probes(ref, n=n, seed=seed, task_name=task)
    else:
        raise ValueError(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, probes)
    return probes


def collect_reps(runs: pd.DataFrame, probes: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    reps: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for r in runs.itertuples():
        reps[r.cond][int(r.seed)] = run_reps(r.run_dir, probes)
    return dict(reps)


def collect_random_init_reps(runs: pd.DataFrame, probes: np.ndarray, n_seeds: int = 5) -> dict[str, dict[int, np.ndarray]]:
    reps: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for cond, grp in runs.groupby("cond"):
        cfg = RunConfig.from_json((Path(grp["run_dir"].iloc[0]) / "config.json").read_text())
        for s in range(n_seeds):
            reps[cond][s] = random_init_reps(cfg, probes, seed=1000 + s)
    return dict(reps)
