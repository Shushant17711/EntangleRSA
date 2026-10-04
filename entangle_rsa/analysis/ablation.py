"""E5 probe-set ablation and E6 observable ablation (Req 6.4, 6.5).

E5: probe size {100, 200, 500, 1000} x sampling {trajectory, uniform}: Spearman(k, RSA_norm) per topology.
E6: the same depth trend computed with zzz-observable quantum reps (single-qubit + all ZZ correlators).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from entangle_rsa.analysis.aggregate import collect_reps, finished_runs, get_probes
from entangle_rsa.rsa.stats import RSAData, trend

REF = "mlp"


def depth_trends(reps: dict, runs: pd.DataFrame) -> dict:
    data = RSAData(reps)
    out = {}
    for topo in ("chain", "all"):
        conds = [(0, "q-sep-k0")] + sorted({(int(k), c) for k, c, t in zip(runs.k, runs.cond, runs.topology) if t == topo})
        conds = [(k, c) for k, c in conds if c in reps]
        if len(conds) < 3:
            continue
        s = [data.summary(c, REF) for _, c in conds]
        ks = [k for k, _ in conds]
        out[topo] = {
            "rsa_norm_trend": trend(ks, [x["rsa_norm"] for x in s]),
            "cka_norm_trend": trend(ks, [x["cka_norm"] for x in s]),
            "rsa_norm": [x["rsa_norm"] for x in s],
        }
    return out


def probe_ablation(root, task) -> pd.DataFrame:
    runs = finished_runs(root, task)
    rows = []
    for kind in ("trajectory", "uniform"):
        for n in (100, 200, 500, 1000):
            probes = get_probes(root, task, runs, kind, n)
            for topo, v in depth_trends(collect_reps(runs, probes), runs).items():
                rows.append({"kind": kind, "n": n, "topology": topo, "rsa_trend": v["rsa_norm_trend"], "cka_trend": v["cka_norm_trend"]})
    return pd.DataFrame(rows)


def observable_ablation(root, task) -> pd.DataFrame:
    zruns = finished_runs(root, task, obs="z")
    zzz = finished_runs(root, task, obs="zzz")
    zzz = pd.concat([zzz[zzz.family == "quantum"], zruns[zruns.family != "quantum"]])
    probes = get_probes(root, task, zruns, "trajectory", 500)
    rows = []
    for name, runs in (("z", zruns), ("zzz", zzz)):
        for topo, v in depth_trends(collect_reps(runs, probes), runs).items():
            rows.append({"observables": name, "topology": topo, "rsa_trend": v["rsa_norm_trend"], "cka_trend": v["cka_norm_trend"],
                         "rsa_norm_by_k": np.round(v["rsa_norm"], 3).tolist()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cartpole")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    out = Path("results") / a.task
    out.mkdir(parents=True, exist_ok=True)
    e5 = probe_ablation(a.root, a.task)
    e5.to_csv(out / "e5_probe_ablation.csv", index=False)
    print(e5.to_markdown(index=False, floatfmt=".3f"))
    e6 = observable_ablation(a.root, a.task)
    e6.to_csv(out / "e6_observable_ablation.csv", index=False)
    print(e6.to_markdown(index=False, floatfmt=".3f"))
