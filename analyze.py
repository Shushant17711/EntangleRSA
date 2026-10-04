"""Full analysis for one task -> results/<task>/ (tables, figures, gates, verdicts).

python analyze.py --task cartpole [--n-boot 2000] [--workers 16] [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entangle_rsa.analysis.aggregate import collect_random_init_reps, collect_reps, finished_runs, get_probes
from entangle_rsa.analysis.gradvar_init import run as gradvar_run
from entangle_rsa.envs.wrappers import get_task, physical_features
from entangle_rsa.rsa.partitioned_analysis import discrepancy, partition_pairs, permutation_test
from entangle_rsa.rsa.stats import RSAData, ci, disagreement_verdict, excludes_zero, hierarchical_bootstrap, piecewise_fit, trend
from entangle_rsa.rsa.transforms import TRANSFORMS

REF = "mlp"


def quantum_conds(runs: pd.DataFrame, topology: str) -> list[tuple[int, str]]:
    q = runs[runs.family == "quantum"]
    out = [(0, "q-sep-k0")] if "q-sep-k0" in set(q.cond) else []
    out += sorted({(int(k), c) for k, c, t in zip(q.k, q.cond, q.topology) if t == topology})
    return out


def depth_curve_stats(rows: pd.DataFrame, conds: list[tuple[int, str]], gv_init: pd.DataFrame | None) -> dict:
    """Per bootstrap resample: trends and breakpoints over k for RSA_norm, CKA_norm, return; then CIs."""
    ks = np.array([k for k, _ in conds], float)
    names = [c for _, c in conds]
    piv = {m: rows.pivot(index="boot", columns="cond", values=m)[names] for m in ("rsa_norm", "cka_norm", "eval_return")}
    out = {}
    for m, P in piv.items():
        vals = P.to_numpy()
        ok = np.isfinite(vals).all(1)
        trends = [trend(ks, v) for v in vals[ok]]
        bps = [piecewise_fit(ks, v)["breakpoint"] for v in vals[ok]] if len(ks) >= 3 else []
        out[m] = {
            "point": [float(np.nanmean(P[c])) for c in names],
            "ci": [ci(P[c]) for c in names],
            "trend_ci": ci(trends),
            "breakpoint_ci": ci(bps) if bps else (np.nan, np.nan),
            "n_valid": int(ok.sum()),
        }
    if gv_init is not None and len(ks) >= 3:
        g = gv_init.set_index("cond").loc[names, "grad_var_mean"].to_numpy()
        lg = np.log(g)
        out["grad_var_init"] = {"point": g.tolist(), "trend": trend(ks, lg), "fit": piecewise_fit(ks, lg)}
    return out


def h1_h2_verdicts(curve: dict) -> dict:
    rsa, ret, cka = curve["rsa_norm"], curve["eval_return"], curve["cka_norm"]
    h1 = "supported" if rsa["trend_ci"][0] > 0 else ("contradicted" if rsa["trend_ci"][1] < 0 else "inconclusive")
    lo = max(rsa["breakpoint_ci"][0], ret["breakpoint_ci"][0])
    hi = min(rsa["breakpoint_ci"][1], ret["breakpoint_ci"][1])
    overlap = bool(np.isfinite(lo) and lo <= hi)
    gv = curve.get("grad_var_init")
    gv_falls = bool(gv and gv["fit"]["slope2"] < 0)
    return {
        "H1": h1,
        "H1_rsa_trend_ci": rsa["trend_ci"],
        "H2_breakpoint_overlap": overlap,
        "H2_gradvar_falls_after_breakpoint": gv_falls,
        "H2": "supported" if (h1 == "supported" and overlap and gv_falls) else "not supported",
        "disagreement": disagreement_verdict(rsa["trend_ci"], cka["trend_ci"]),
        "cka_trend_ci": cka["trend_ci"],
    }


def invariance_table(reps: dict, conds: list[str], ref: str = REF) -> pd.DataFrame:
    rows = []
    for name, fn in TRANSFORMS.items():
        data = RSAData({c: {s: fn(r) for s, r in reps[c].items()} for c in conds + [ref]})
        for c in conds:
            s = data.summary(c, ref)
            rows.append({"transform": name, "cond": c, "rsa_norm": s["rsa_norm"], "cka_norm": s["cka_norm"], "rsa": s["rsa"], "cka": s["cka"]})
    return pd.DataFrame(rows)


def h3(reps: dict, probes: np.ndarray, task_name: str, topology: str, n_perm: int) -> list[dict]:
    task = get_task(task_name)
    feats = physical_features(task, probes)
    part = partition_pairs(feats, task.coupled_dims, task.uncoupled_dims)
    out = []
    dense_cond = f"q-{topology}-k4"
    if "q-sep-k0" not in reps or dense_cond not in reps:
        return out
    seeds = sorted(set(reps["q-sep-k0"]) & set(reps[dense_cond]) & set(reps[REF]))
    for s in seeds:
        d_sep = discrepancy(reps["q-sep-k0"][s], reps[REF][s])
        d_dense = discrepancy(reps[dense_cond][s], reps[REF][s])
        res = permutation_test(d_sep, d_dense, part, n_perm=n_perm, seed=s)
        out.append({"seed": s, "topology": topology, **res, "n_coupled": int((part.labels == 1).sum()), "n_uncoupled": int((part.labels == 0).sum())})
    return out


def plot_main(curves: dict, gv: pd.DataFrame | None, path: Path, task: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {"chain": "#1f77b4", "all": "#d62728"}
    for topo, (conds, curve) in curves.items():
        ks = [k for k, _ in conds]
        for ax, m, lab in ((axes[0], "rsa_norm", "RSA (noise-ceiling normalised)"), (axes[1], "eval_return", "eval return")):
            pt = curve[m]["point"]
            lo, hi = zip(*curve[m]["ci"])
            ax.plot(ks, pt, "o-", color=colors[topo], label=f"{topo}")
            ax.fill_between(ks, lo, hi, color=colors[topo], alpha=0.2)
            ax.set_ylabel(lab)
        cpt = curve["cka_norm"]["point"]
        axes[0].plot(ks, cpt, "s--", color=colors[topo], alpha=0.6, label=f"{topo} (CKA)")
        if gv is not None:
            g = gv[gv.cond.isin([c for _, c in conds])].set_index("cond").loc[[c for _, c in conds]]
            axes[2].semilogy(ks, g.grad_var_mean, "o-", color=colors[topo], label=topo)
    axes[2].set_ylabel("Var ∂C/∂θ at init")
    for ax in axes:
        ax.set_xlabel("entangling depth k")
        ax.legend(fontsize=8)
    fig.suptitle(f"{task}: representation, return and trainability vs entangling depth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_h3(h3_rows: list[dict], path: Path, task: str):
    if not h3_rows:
        return
    df = pd.DataFrame(h3_rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, (topo, g) in enumerate(df.groupby("topology")):
        ax.scatter(np.full(len(g), i) + np.linspace(-0.1, 0.1, len(g)), g["T"], label=topo)
        ax.errorbar(i + 0.25, g["T"].mean(), yerr=g["null_sd"].mean() * 1.96, fmt="k_", capsize=4)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(df.topology.nunique()), sorted(df.topology.unique()))
    ax.set_ylabel("T = coupled − uncoupled\n(D_sep − D_dense)")
    ax.set_title(f"{task}: H3 localised dissimilarity (per seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cartpole")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--n-probes", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="small bootstrap/permutation counts for a smoke run")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_boot, a.n_perm = 100, 500

    out = Path(a.out) / a.task
    out.mkdir(parents=True, exist_ok=True)
    runs = finished_runs(a.root, a.task)
    if runs.empty:
        raise SystemExit(f"no finished runs for {a.task}")
    runs.to_csv(out / "runs.csv", index=False)
    probes = get_probes(a.root, a.task, runs, "trajectory", a.n_probes)
    reps = collect_reps(runs, probes)
    conds_all = sorted(c for c in reps if c != REF)
    per_seed = {c: {"eval_return": [runs[(runs.cond == c) & (runs.seed == s)].eval_return.iloc[0] for s in sorted(reps[c])]} for c in reps}

    print(f"[{a.task}] {len(runs)} runs, {len(conds_all)} conditions, {len(probes)} probes; bootstrap {a.n_boot}", flush=True)
    data = RSAData(reps)
    point = pd.DataFrame([{"cond": c, **data.summary(c, REF)} for c in conds_all])
    rows = pd.DataFrame(hierarchical_bootstrap(data, conds_all, REF, per_seed, a.n_boot, workers=a.workers))
    rows.to_csv(out / "bootstrap.csv.gz", index=False)

    gv = None
    if get_task(a.task).obs_dim <= 6:
        gv = pd.DataFrame(gradvar_run(a.task, probes, out / "gradvar_init.csv", n_inits=200 if not a.quick else 30))

    curves, verdicts = {}, {}
    for topo in ("chain", "all"):
        conds = quantum_conds(runs, topo)
        if len(conds) < 2:
            continue
        curve = depth_curve_stats(rows, conds, gv)
        curves[topo] = (conds, curve)
        verdicts[topo] = h1_h2_verdicts(curve) if len(conds) >= 3 else {"note": "fewer than 3 depths; trend only", "rsa_trend_ci": curve["rsa_norm"]["trend_ci"]}

    # random-init baseline (capacity/architecture) and invariance diagnostics
    rinit = collect_random_init_reps(runs, probes)
    rdata = RSAData(rinit)
    rinit_tab = pd.DataFrame([{"cond": c, **{f"init_{k}": v for k, v in rdata.summary(c, REF).items()}} for c in conds_all])
    inv = invariance_table(reps, conds_all)
    inv.to_csv(out / "invariance.csv", index=False)

    h3_rows = []
    for topo in ("chain", "all"):
        h3_rows += h3(reps, probes, a.task, topo, a.n_perm)
    pd.DataFrame(h3_rows).to_csv(out / "h3.csv", index=False)

    # uniform-probe robustness (E5 subset; full ablation in analysis/ablation)
    uprobes = get_probes(a.root, a.task, runs, "uniform", a.n_probes)
    udata = RSAData(collect_reps(runs, uprobes))
    uni = pd.DataFrame([{"cond": c, **{f"uni_{k}": v for k, v in udata.summary(c, REF).items() if k in ("rsa_norm", "cka_norm")}} for c in conds_all])

    ret = runs.groupby("cond").agg(
        eval_return_mean=("eval_return", "mean"), eval_return_std=("eval_return", "std"), n_seeds=("seed", "count"),
        train_minutes=("train_seconds", lambda s: s.mean() / 60),
    )
    cis = rows.groupby("cond").agg(
        rsa_norm_lo=("rsa_norm", lambda v: ci(v)[0]), rsa_norm_hi=("rsa_norm", lambda v: ci(v)[1]),
        cka_norm_lo=("cka_norm", lambda v: ci(v)[0]), cka_norm_hi=("cka_norm", lambda v: ci(v)[1]),
        nan_resamples=("rsa_norm", lambda v: int(np.isnan(v).sum())),
    )
    table = point.set_index("cond").join(cis).join(ret).join(rinit_tab.set_index("cond")).join(uni.set_index("cond"))
    if gv is not None:
        table = table.join(gv.set_index("cond")[["grad_var_mean"]].rename(columns={"grad_var_mean": "grad_var_init"}))
    table = pd.concat([table, ret.loc[[c for c in ret.index if c not in table.index]]])
    table.to_csv(out / "main_table.csv")
    cols = ["n_seeds", "eval_return_mean", "eval_return_std", "rsa", "rsa_ceil", "rsa_norm", "rsa_norm_lo", "rsa_norm_hi",
            "cka", "cka_ceil", "cka_norm", "init_rsa_norm", "init_cka_norm", "uni_rsa_norm", "grad_var_init"]
    (out / "main_table.md").write_text(table[[c for c in cols if c in table.columns]].to_markdown(floatfmt=".3f"))

    plot_main(curves, gv, out / "rsa_return_gradvar_curve.png", a.task)
    plot_h3(h3_rows, out / "partitioned_dissimilarity.png", a.task)

    summary = {
        "task": a.task,
        "n_runs": len(runs),
        "n_boot": a.n_boot,
        "probe_kind": "trajectory (pooled, equal share per run)",
        "verdicts": verdicts,
        "h3": {
            topo: {
                "mean_T": float(np.mean([r["T"] for r in h3_rows if r["topology"] == topo])) if any(r["topology"] == topo for r in h3_rows) else None,
                "n_seeds_p<0.05": int(sum(r["p"] < 0.05 for r in h3_rows if r["topology"] == topo)),
                "n_seeds": int(sum(r["topology"] == topo for r in h3_rows)),
            }
            for topo in ("chain", "all")
        },
        "ceilings": {c: {"rsa": float(point.set_index("cond").loc[c, "rsa_ceil"])} for c in conds_all},
        "ref_ceiling_rsa": float(point["rsa_ref_ceil"].iloc[0]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["verdicts"], indent=1, default=float))
    print((out / "main_table.md").read_text())


if __name__ == "__main__":
    main()
