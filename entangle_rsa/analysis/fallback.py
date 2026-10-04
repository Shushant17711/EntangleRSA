"""Pre-registered Pendulum fallback (Req 6.2): exit code 1 if triggered.

Trigger: the k=0 and k=4 (chain) quantum agents do not reach mean return >= -400 within 300k steps in
at least 3 of 5 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THRESH, STEPS, MIN_SEEDS = -400.0, 300_000, 3


def seed_reached(run_dir: Path) -> bool:
    m = pd.read_csv(run_dir / "metrics.csv")
    m = m[m.global_step <= STEPS]
    return bool(len(m) and m.ep_return_mean.max() >= THRESH)


def check(root: Path | str, task: str = "pendulum") -> dict:
    out = {}
    for cond in ("q-sep-k0-z", "q-chain-k4-z"):
        dirs = sorted(Path(root, task, cond).glob("seed[0-4]"))
        hits = [seed_reached(d) for d in dirs if (d / "done").exists()]
        out[cond] = {"n_seeds": len(hits), "n_reached": int(sum(hits))}
    out["triggered"] = any(v["n_reached"] < MIN_SEEDS for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pendulum")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    res = check(a.root, a.task)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/fallback_{a.task}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(1 if res["triggered"] else 0)
