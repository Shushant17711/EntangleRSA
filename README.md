# EntangleRSA — what does entanglement buy a quantum RL policy?

We train parameterised-quantum-circuit (PQC) PPO policies at different **entangling depths**
k ∈ {0..4} (CZ entanglers after the first k of L = 4 variational layers; `chain` and all-to-all `all`
topologies, identical parameter count). We compare their internal representations against classical MLP
policies using **representational similarity analysis (RSA)** and linear CKA, with noise ceilings and a
hierarchical bootstrap.

Pre-registered hypotheses (brief §4, `.kiro/specs/entangle-rsa/requirements.md`):

- **H1**: quantum–classical RSA increases with entangling depth up to a saturation point, then plateaus.
- **H2**: the RSA saturation point coincides with the return saturation point. Depth beyond it costs
  trainability (lower gradient variance).
- **H3**: separable circuits differ most from the MLP on coupled degrees of freedom.

## Results

<!-- RESULTS:BEGIN -->
All numbers come from `results/` (`<task>/main_table.md`, `<task>/summary.json`, figures, `gates.md`).
Sweeps: CartPole 70 runs, Pendulum 70, Acrobot 70, CartPole `zzz` ablation 45. Every run finished; none failed.
Analysis: 500 pooled trajectory probes, 2000 hierarchical bootstrap resamples.

### Gates (pre-registered)

| Gate | Outcome |
|---|---|
| G1 E1 measurement validation | **PASS** |
| G2 quantum PPO solves CartPole | **PASS**: `q-sep-k0` and `q-chain-k4` 5/5 seeds at 500 |
| G3 quantum seed-to-seed RSA ceiling ≥ 0.2 | **FAIL**: `q-all-k3` on CartPole = 0.191 |

Because G3 failed, the pre-registered consequence applies: **representational reliability is the headline
result**, and every RSA_norm below has to be read alongside its ceiling.

### Headline: quantum policy representations are not very reliable

Seed-to-seed RSA ceilings (agreement between two seeds of the *same* condition):

| Task | Quantum ceilings | Primary MLP ceiling |
|---|---|---|
| CartPole | 0.19 – 0.60 | 0.49 |
| Pendulum | 0.27 – 0.58 | 0.73 |
| Acrobot | 0.26 – 0.76 | 0.84 |

Two seeds of the same circuit often agree less with each other than with the MLP. The ceilings tend to
be lowest for deep all-to-all circuits (CartPole `q-all-k3` 0.19, Pendulum `q-all-k2` 0.27, Acrobot `q-all-k4` 0.26).
Normalisation by such small ceilings makes RSA_norm noisy, and some bootstrap intervals exceed 1.

### Pre-registered hypotheses

| | CartPole (primary) | Acrobot (fallback 2nd task) | Pendulum (H3 only) |
|---|---|---|---|
| **H1** RSA rises with depth | inconclusive (chain CI [−0.6, 0.9]; all [0.0, 0.9]) | chain **supported**, CI [0.1, 1.0]; all inconclusive [−0.3, 1.0] | — |
| **H2** RSA saturation = return saturation | not supported (return at ceiling) | chain supported; all not supported | — |
| **H3** separable circuits differ on coupled pairs | not supported (3/10 chain, 2/5 all seeds p<0.05; sign varies) | 8/10 chain, 3/5 all seeds p<0.05; small T (0.009 / 0.002) | **supported**: 10/10 chain, 4/5 all seeds; T = 0.067 / 0.055 |

- The Pendulum fallback triggered (`results/fallback_pendulum.json`): `q-sep-k0` reached −400 in 0/5 seeds,
  so per the pre-registration Pendulum is used only for H3 and Acrobot is the second depth-curve task.
- The Acrobot H1 support is weak. Chain RSA_norm is flat at 0.82–0.87 for k=0–3 and only jumps at k=4 (0.96).
- CKA agrees in direction with RSA on every task and topology (no "metric-dependent" verdicts).

### Findings that hold everywhere

- **Gradient variance at initialisation falls with entangling depth** on all three tasks, and faster for
  `all` than for `chain`. For example, on Acrobot it falls from 0.025 at k=0 to 0.008 (`chain` k=4)
  and 0.002 (`all` k=4). This is the barren-plateau proxy behaving as expected.
- **Training** (eval return, mean ± std over seeds):
  - CartPole: every condition reaches 500 except `q-all-k2` (449.5 ± 112.9) and `q-all-k4` (443.8 ± 125.6).
  - Acrobot: quantum conditions reach about −82 to −95, matching the MLP (−85). `q-all-k4` (−333) and `mlp-pm` (−285) have failed seeds.

### Exploratory (not pre-registered, Pendulum)

- Under the frozen hyperparameters, **only entangled circuits solve Pendulum**:
  k ≥ 3 reaches about −150 on both topologies, and `q-all-k2` reaches −167.
  k = 0, k = 1 and `q-chain-k2` stay at −670 to −860, and the primary MLP fails too (−629 ± 145).
  This is a hyperparameter-specific result: the PPO settings were tuned only on CartPole.
- Because the reference MLP fails Pendulum, Pendulum's RSA-vs-MLP numbers (and the H3 effect) compare circuits
  against a non-solving policy. The H3 result is still the strongest signal in the study, but this caveat is in LIMITATIONS.

### Ablations (CartPole)

- **E6 observables** (`results/cartpole/e6_observable_ablation.csv`): measuring ⟨Z⟩+⟨ZZ⟩ instead of ⟨Z⟩
  lowers RSA_norm at every depth (k=0: 0.57 → 0.36). The depth trend stays positive with both observable
  sets (point Spearman 0.6–1.0). These are point estimates without bootstrap intervals.
- **E5 probes** (`results/cartpole/e5_probe_ablation.csv`): the point trend is stable from 200 to 1000 probes
  for both sampling schemes. It drops on `chain` at 100 trajectory probes (0.2). Uniform probes give a slightly stronger `chain` trend (0.9).
<!-- RESULTS:END -->

## Setup

```bash
uv venv .venv && uv pip install -e '.[dev]'   # Python ≥ 3.11
.venv/bin/python -m pytest -q                 # fast tests
.venv/bin/python -m pytest -q -m ""           # + slow E1 validation suite (gate G1)
```

## Reproduce

```bash
WORKERS=16 bash scripts/reproduce.sh
```

The script runs these steps in order. It aborts if G1 fails, and it skips completed runs (`done` marker),
so you can re-run it after an interruption.

1. **G1**: the full test suite, including the E1 measurement-validation suite and the statevector oracle.
2. **E2**: CartPole sweep (70 runs: 9 quantum conditions × 5 seeds, `mlp` and `mlp-pm` × 5, plus extra
   seeds to 10 for `q-sep-k0`, `q-chain-k4`, `mlp`), followed by analysis.
3. **E3**: Pendulum sweep. The pre-registered fallback check runs first; if it triggers, an Acrobot sweep
   becomes the second task and Pendulum is used only for H3.
4. **E6/E5**: ⟨Z⟩+⟨ZZ⟩ observable ablation on CartPole, and the probe-size × sampling ablation.
5. **Gates**: `results/gates.md`.

A full run takes about 10–12 h on 16 CPU workers (a quantum CartPole run is ~30 min/worker). To run a
single configuration:

```bash
.venv/bin/python train.py --help
.venv/bin/python sweep.py e2 --dry-run
```

## Matching criteria (Req 2.3)

- **Quantum**: n_qubits = observation dim, data re-uploading with trainable input scaling, RY/RZ layers.
  The representation is ⟨Z_i⟩ (ablation: + ⟨Z_iZ_j⟩).
- **`mlp`** (primary): obs → 64 (tanh) → n_qubits (tanh) → head. It matches representation width and
  bounded range.
- **`mlp-pm`** (secondary): same shape, with the hidden width chosen so the encoder parameter count matches
  the circuit (±10%).
- The policy heads and the critic MLP are identical across all conditions.
- Frozen PPO hyperparameters (`entangle_rsa/config.py`): actor LR 5e-4, quantum LR 5e-3, critic LR 2.5e-3,
  500k steps (CartPole). The history behind these values is in `results/pilot.md`.

## Layout

```
entangle_rsa/policies/   circuits (PennyLane), statevector oracle, quantum/classical actors, heads
entangle_rsa/rl/ppo.py   PPO with per-minibatch actor gradient-variance logging
entangle_rsa/rsa/        RDMs, RSA/CKA, transforms, probes, extraction, stats, H3 partitioned analysis
entangle_rsa/analysis/   init grad variance, aggregation, fallback check, ablations, gates
train.py / sweep.py / analyze.py   CLIs
results/                 figures, tables, gates (generated)
```

See `LIMITATIONS.md` for what this does and does not show.

## License

MIT. See [LICENSE](LICENSE).
