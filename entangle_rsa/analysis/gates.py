"""Report the pre-registered go/no-go gates (design §12) -> results/gates.md. Failures change the headline, never silently."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from entangle_rsa.analysis.aggregate import finished_runs


def g1() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "", "tests/test_validation_e1.py", "tests/test_quantum_circuit.py"],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:]


def g2(root="runs") -> tuple[bool, str]:
    runs = finished_runs(root, "cartpole")
    msgs, ok = [], True
    for cond in ("q-sep-k0", "q-chain-k4"):
        r = runs[(runs.cond == cond) & (runs.seed < 5)]
        n = int((r.eval_return >= 475).sum())
        ok &= n >= 3
        msgs.append(f"{cond}: {n}/{len(r)} seeds eval return ≥ 475")
    return ok, "; ".join(msgs)


def g3(task="cartpole") -> tuple[bool, str]:
    p = Path("results") / task / "summary.json"
    if not p.exists():
        return False, "analysis not run"
    s = json.loads(p.read_text())
    q = {c: v["rsa"] for c, v in s["ceilings"].items() if c.startswith("q-")}
    bad = {c: round(v, 3) for c, v in q.items() if not v >= 0.2}
    return not bad, f"min quantum seed-to-seed RSA ceiling = {min(q.values()):.3f}" + (f"; below 0.2: {bad}" if bad else "")


def main():
    lines = ["# Gates (pre-registered, design §12)", ""]
    for name, (ok, msg) in (("G1 E1 measurement validation", g1()), ("G2 quantum PPO solves CartPole", g2()), ("G3 quantum ceiling ≥ 0.2", g3())):
        lines.append(f"- **{name}: {'PASS' if ok else 'FAIL'}** — {msg}")
    lines += ["", "Consequences of failure: G1 → no experiment results are reported; G2 → RL sweep scope is cut and",
              "the return axis reported with that caveat; G3 → representational reliability becomes the headline result."]
    Path("results").mkdir(exist_ok=True)
    Path("results/gates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
