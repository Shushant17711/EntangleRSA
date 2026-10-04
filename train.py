"""Train one run: python train.py --task cartpole --family quantum --topology chain --k 2 --seed 0"""

from __future__ import annotations

import argparse

import torch

from entangle_rsa.config import RunConfig, task_defaults
from entangle_rsa.rl.ppo import train


def parse(argv=None) -> tuple[RunConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cartpole")
    ap.add_argument("--family", default="quantum", choices=["quantum", "mlp", "mlp-pm"])
    ap.add_argument("--topology", default="chain", choices=["chain", "all"])
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--obs", default="z", choices=["z", "zzz"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="override total env steps")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    ppo = task_defaults(args.task)
    if args.steps is not None:
        ppo.total_steps = args.steps
    cfg = RunConfig(task=args.task, family=args.family, topology=args.topology, k=args.k, obs=args.obs, seed=args.seed, ppo=ppo)
    return cfg, args


def main(argv=None):
    cfg, args = parse(argv)
    torch.set_num_threads(args.threads)
    out = train(cfg, root=args.root, verbose=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
