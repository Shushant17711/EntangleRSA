# Limitations

Written before final results came in and updated afterwards. Anything that changed after results is
marked *(post-results)*.

## Scope
- **Classical simulation only.** Circuits are simulated exactly with PennyLane `default.qubit`, with no
  shot noise or hardware noise. Nothing here says anything about NISQ hardware.
- **Small systems.** We use 4 qubits (CartPole/Acrobot-sized) and 3 qubits (Pendulum), with L = 4 layers,
  so entangling depth k ∈ {0..4}. That gives five points on the depth axis. A "saturation point" fit to
  five points is coarse, and the piecewise-fit breakpoint CI is correspondingly wide.
- **Environments.** CartPole-v1 is the primary task. It is near-trivially solvable, so return saturates for
  every condition that trains at all. The return axis of H2 therefore has very little dynamic range on
  CartPole.
- **One entangler family.** Only parameter-free CZ entanglers in `chain` and `all` topologies were tested.
  Other gates (CNOT, CRZ, trainable entanglers) were not.

## Matching quantum and classical policies
- The primary MLP matches the **representation shape and range** (width = n_qubits, tanh-bounded, like
  ⟨Z_i⟩) but has more parameters. The secondary `mlp-pm` matches **parameter count** (±10%) instead.
  Neither is a uniquely "fair" comparator, so both are reported.
- Heads and the critic are identical classical modules across families, so only the encoder differs.
  Policy behaviour still differs, though, so probe states come from different state distributions. Pooling
  probes with an equal share per run reduces this but cannot remove it.

## Hyperparameters
- Learning rates were tuned only in the pilot (three rounds, documented in `results/pilot.md` and the
  design.md changelog) and then frozen. The critic LR fix (2.5e-3) was found with an MLP-only diagnostic
  and applied to all families. No per-condition tuning was done, so conditions that would train with a
  different LR may be disadvantaged.
- Training is 500k environment steps per run. Conclusions about "beyond saturation" cost hold only at this
  budget.

## Measurement
- RSA/CKA compare representational geometry over a finite probe set (default 500 states). The E5 probe
  ablation reports sensitivity to probe count and sampling scheme.
- Noise ceilings are estimated from 5 seeds (10 for selected conditions). Low ceilings make RSA_norm
  unstable, which is why gate G3 exists.
- The gradient-variance trainability proxy is measured both at init (200 random inits) and during training.
  It is a proxy for barren plateaus, not a proof of their presence.

## Statistics
- Hypotheses H1–H3 were pre-registered (requirements.md §6–7, brief §4) and their wording was not edited
  after results came in.
- The Spearman(k, RSA_norm) trend uses only five depth values per topology. The bootstrap CI accounts for
  seed and probe variability but not for the choice of depth grid.

## Added after results came in *(post-results)*
- **G3 failed** on CartPole (`q-all-k3` ceiling 0.191 < 0.2). Low ceilings (down to 0.26–0.27 on Acrobot and
  Pendulum too) make RSA_norm noisy, and some bootstrap CIs exceed 1. Per the pre-registration, reliability is
  the headline result.
- **Pendulum fallback triggered.** `q-sep-k0` never reached −400, so Pendulum is used only for H3. Acrobot
  (6 qubits) replaced it as the second depth-curve task. Acrobot circuits are larger than CartPole's,
  so depth effects across the two tasks are confounded with qubit count.
- **The Pendulum reference MLP fails the task** (−629 ± 145) under the frozen CartPole-tuned hyperparameters, as do
  shallow circuits. The Pendulum H3 contrast therefore compares circuits with a non-solving reference policy.
  The finding that "entangled circuits solve Pendulum and the MLP doesn't" is exploratory and specific to these hyperparameters.
- **The Acrobot H1 support is driven by one point.** Chain RSA_norm is flat for k=0–3, and the rise comes from k=4.
- **The E5/E6 ablations report point Spearman trends without bootstrap CIs.**
- **The analysis bootstrap uses one BLAS thread per worker.** This is a performance fix (`scripts/reproduce.sh`). It does not change results,
  because the bootstrap is seeded per resample and is independent of worker count (`tests/test_stats.py`).
