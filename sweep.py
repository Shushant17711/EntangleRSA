"""Run a named grid of training runs in parallel processes; completed runs (`done` marker) are skipped.

python sweep.py pilot | e2 | e3 | e3-acrobot | ablation-zzz  [--workers 16] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from entangle_rsa.config import N_LAYERS, RunConfig, task_defaults

QUANTUM_CONDS = [("chain", 0)] + [(t, k) for t in ("chain", "all") for k in range(1, N_LAYERS + 1)]


def _q(task, topo, k, seed, obs="z"):
    return RunConfig(task=task, family="quantum", topology=topo, k=k, obs=obs, seed=seed, ppo=task_defaults(task))


def _c(task, family, seed):
    return RunConfig(task=task, family=family, seed=seed, ppo=task_defaults(task))


def sweep_grid(task: str, seeds=range(5), ceiling_seeds=range(5, 10)) -> list[RunConfig]:
    cfgs = [_q(task, t, k, s) for t, k in QUANTUM_CONDS for s in seeds]
    cfgs += [_c(task, f, s) for f in ("mlp", "mlp-pm") for s in seeds]
    # extra seeds for noise ceilings (council D6)
    cfgs += [_q(task, "chain", 0, s) for s in ceiling_seeds]
    cfgs += [_q(task, "chain", N_LAYERS, s) for s in ceiling_seeds]
    cfgs += [_c(task, "mlp", s) for s in ceiling_seeds]
    return cfgs


GRIDS = {
    "pilot": lambda: [_q("cartpole", "chain", 0, s) for s in range(3)]
    + [_q("cartpole", "chain", N_LAYERS, s) for s in range(3)]
    + [_c("cartpole", "mlp", s) for s in range(3)],
    "e2": lambda: sweep_grid("cartpole"),
    "e3": lambda: sweep_grid("pendulum"),
    "e3-acrobot": lambda: sweep_grid("acrobot"),
    "ablation-zzz": lambda: [_q("cartpole", t, k, s, obs="zzz") for t, k in QUANTUM_CONDS for s in range(5)],
}


def _worker(args):
    cfg_json, root = args
    import torch

    torch.set_num_threads(1)
    from entangle_rsa.rl.ppo import train

    cfg = RunConfig.from_json(cfg_json)
    t0 = time.time()
    try:
        train(cfg, root=root)
        return cfg.run_dir(root).as_posix(), "ok", time.time() - t0
    except Exception:
        err = traceback.format_exc()
        d = cfg.run_dir(root)
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(err)
        return d.as_posix(), "error", time.time() - t0


def run_grid(cfgs: list[RunConfig], root: str = "runs", workers: int = 16) -> list[tuple]:
    todo = [c for c in cfgs if not (c.run_dir(root) / "done").exists()]
    print(f"{len(cfgs)} runs in grid, {len(cfgs) - len(todo)} done, {len(todo)} to run on {workers} workers", flush=True)
    # longest (most entangling) first for better packing
    todo.sort(key=lambda c: (c.family != "quantum", -c.k))
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, [(c.to_json(), root) for c in todo]), 1):
            results.append(res)
            print(f"[{i}/{len(todo)}] {res[1]} {res[0]} ({res[2] / 60:.1f} min)", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=sorted(GRIDS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfgs = GRIDS[args.grid]()
    if args.dry_run:
        for c in cfgs:
            print(c.run_dir(args.root))
        print(len(cfgs), "runs")
        return
    res = run_grid(cfgs, args.root, args.workers)
    failed = [r for r in res if r[1] != "ok"]
    Path(args.root).mkdir(exist_ok=True)
    (Path(args.root) / f"sweep_{args.grid}_log.json").write_text(json.dumps(res, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {[f[0] for f in failed]}")


if __name__ == "__main__":
    main()
