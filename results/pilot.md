# CartPole pilot (task 6.2)

Pilot grid: `mlp`, `q-sep-k0-z`, `q-chain-k4-z` × seeds 0–2, 500k env steps, frozen defaults.

## History
- **Pilot 1** (actor/quantum LR amendment, see design.md changelog): quantum conditions trained;
  MLP returns unstable (eval 391 ± 189). Artifacts: `runs_archive/cartpole_pilot1`, `runs_archive/results_pilot1`.
- **Pilot 2** (actor 5e-4 / quantum 5e-3 / critic 5e-4): aborted — MLP collapsed after reaching ~500
  despite small KL. An MLP-only diagnostic (scratch, not reported) isolated the cause to the critic
  lagging the policy; critic LR 2.5e-3 fixed it. The critic is shared across families, so this is a shared fix.
  Artifacts: `runs_archive/cartpole_pilot2_aborted`.
- **Pilot 3** (FINAL frozen defaults: lr actor 5e-4, quantum 5e-3, critic 2.5e-3, 500k steps): below.
  No further tuning is permitted regardless of later results.

## Pilot 3 results

| condition | seed | eval return (mean ± std) | train time |
|---|---|---|---|
| mlp | 0 | 500.0 ± 0.0 | 2.2 min |
| mlp | 1 | 500.0 ± 0.0 | 1.8 min |
| mlp | 2 | 500.0 ± 0.0 | 1.7 min |
| q-sep-k0-z | 0 | 500.0 ± 0.0 | 27.0 min |
| q-sep-k0-z | 1 | 500.0 ± 0.0 | 24.4 min |
| q-sep-k0-z | 2 | 500.0 ± 0.0 | 28.6 min |
| q-chain-k4-z | 0 | 500.0 ± 0.0 | 34.3 min |
| q-chain-k4-z | 1 | 500.0 ± 0.0 | 32.6 min |
| q-chain-k4-z | 2 | 500.0 ± 0.0 | 34.8 min |

**G2: PASS** — every pilot seed of both quantum conditions reaches eval return ≥ 475 (all 500).
These runs are reused as seeds 0–2 of the E2 sweep.
