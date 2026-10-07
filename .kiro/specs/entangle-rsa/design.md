# Design — EntangleRSA

## §1 Overview
Train PPO agents whose actor is either a PQC (varying entangling depth k and topology) or a classical
MLP, then compare their internal representations with RSA (primary) and linear CKA (secondary) over a
shared probe set of visited states. Everything runs on one laptop (RTX 4060, 20 cores); simulation is
small (3–6 qubits) so runs are CPU-parallel, one process per run.

```
envs/ (obs scaling) ─► policies/ (PQC | MLP actor, shared critic) ─► rl/ppo.py ─► runs/<task>/<cond>/seed<s>/
                                                                                  │ model.pt, metrics.csv, config.json
rsa/probes.py (pooled visited states) ─► rsa/extract.py (reps) ─► rsa/core.py, rsa/stats.py, rsa/partitioned_analysis.py
                                                                                  ▼
                                                          analysis/ figures + results/main_table.md
```

## §2 Package layout
```
entangle_rsa/
  config.py               # RunConfig dataclass, condition naming, (de)serialisation
  envs/wrappers.py        # make_env(task): fixed observation scaling; task metadata (coupled dims)
  policies/circuits.py    # circuit builder: layer structure, entangler pairs, observables (PennyLane)
  policies/statevector.py # independent torch statevector oracle (tests only)
  policies/quantum_policy.py   # QuantumActor(nn.Module): forward(obs)->rep; rep->head
  policies/classical_policy.py # MLPActor (primary width-matched / param-matched), Critic
  policies/heads.py       # CategoricalHead, GaussianHead
  rl/ppo.py               # PPO trainer (vector envs), grad-variance logging
  rsa/core.py             # rdm, spearman, rsa_score, cka_linear (DONE)
  rsa/transforms.py       # zscore, pca_whiten, rank_transform
  rsa/validation_toy_task.py   # E1
  rsa/probes.py           # probe collection: trajectory-pooled, uniform
  rsa/extract.py          # load checkpoint -> representation matrix
  rsa/stats.py            # ceilings, hierarchical bootstrap, piecewise fit, permutation test
  rsa/partitioned_analysis.py  # H3
  analysis/gradvar_init.py     # E6 init-gradient variance
  analysis/aggregate.py   # builds tables and figures
train.py                  # CLI: one run
sweep.py                  # CLI: grid of runs, process pool, skip done
analyze.py                # CLI: full analysis → results/
scripts/reproduce.sh
```

## §3 Quantum circuit (Req 1.1–1.5)
n = obs dim qubits, L = 4. Inputs x̃ = scaled observation (§6). For layer l = 0..L−1:
1. Variational block: RY(θ_{l,q,0}) RZ(θ_{l,q,1}) on every qubit q.
2. If l < k: CZ on every pair in topology (`chain`: (q,q+1); `all`: all q<r).
3. Encoding block: RX(λ_{l,q} · x̃_q) on qubit q (λ init 1, trainable).
Final variational block (l = L): RY, RZ on every qubit. Then measure observables.
Parameters: 2n(L+1) + nL = 56 for n=4 (independent of k, topology — CZ is parameter-free).
Condition name: `q-{topo}-k{k}` (k=0 → `q-sep-k0`, shared by both topologies). Observables: `z` (n values) or `zzz` (n + n(n−1)/2).
Separability guarantee: with k=0 each qubit only ever sees its own x̃_q ⇒ ∂⟨Z_i⟩/∂x_j = 0 for j≠i (test).
Implementation: PennyLane `default.qubit`, torch interface, parameter broadcasting over the batch dim.
Oracle: `statevector.py` implements the same circuit by tensor contraction; tests compare outputs and gradients.

## §4 Actors, critic, heads (Req 2, 3.1)
- QuantumActor: rep = circuit(x̃) ∈ [−1,1]^d; head = Linear(d, A) (discrete logits) or Linear(d,1) mean + state-independent log-std (Gaussian).
- MLPActor primary: x̃ → Linear(n,64) → tanh → Linear(64,n) → tanh = rep → same head. Secondary (`mlp-pm`): x̃ → Linear(n,h) → tanh → Linear(h,n) → tanh with h chosen so actor params ≈ quantum actor params (±10%).
- Critic (all conditions identical): x̃ → 64 → tanh → 64 → tanh → 1.
- Representation used by RSA = `rep` for both families (pre-head).

## §5 PPO (Req 3.1, 3.3, 3.4)
CleanRL-style: 8 sync vector envs, 128 steps/rollout, GAE(γ=0.99, λ=0.95), 4 epochs, 4 minibatches, clip 0.2,
entropy 0.01 (discrete) / 0.0 (Gaussian), vf coef 0.5, max-grad-norm 0.5, Adam.
LR: classical params 2.5e-3 for actor heads/MLP, 1e-3 critic; quantum angles 1e-2 and λ 1e-2 (tuned once in the
pilot, then frozen in config defaults — never tuned per condition). Steps: CartPole 300k, Pendulum 500k (Gaussian,
γ=0.9 per common Pendulum PPO settings, 16 envs ×  64 steps → tuned in pilot), Acrobot 300k.
Grad variance: during each update, per-minibatch actor-gradient vectors are collected; log mean over parameters of
Var_minibatch[∂L/∂θ] for the actor (quantum: circuit params only; classical: all actor params).
Run dir `runs/{task}/{cond}-{obs}/seed{s}/` contains `config.json`, `metrics.csv` (update, step, ep_return_mean,
grad_var, losses), `model.pt`, `done` marker. Eval: 20 deterministic episodes at end → `eval.json`.

## §6 Environments (Req 3.2)
`make_env(task)` returns a gymnasium env with a fixed observation transform x̃ = obs / scale (no clipping), scales: CartPole (2.4, 3.0, 0.21, 3.0); Pendulum (1, 1, 8); Acrobot (1,1,1,1, 4π, 9π).
TaskSpec also records `coupled_dims` and `uncoupled_dims` for H3:
- CartPole: coupled (θ=2, ẋ=1), uncoupled (x=0, θ̇=3).
- Pendulum: coupled via derived features (angle = atan2(sin,cos), θ̇); uncoupled control = a shuffled-feature pairing (documented in LIMITATIONS: Pendulum has only one physical pair, so the H3 contrast is angle-vs-velocity-only pairs vs joint pairs).

## §7 Probes (Req 4.2)
`collect_trajectory_probes(task, run_dirs, n=500, seed)`: roll out each trained run's deterministic+stochastic
policy for enough episodes, pool states per run, take an equal share per run, then subsample to n with a fixed
RNG. Saved to `runs/{task}/probes_{kind}_{n}.npy` (raw obs) so every model sees identical probes.
`uniform_probes(task, n, seed)`: uniform in the box spanned by the 1st–99th percentile of the trajectory pool.

## §8 RSA statistics (Req 4.4–4.6, 7)
- RDM metric default: correlation distance (spec §5.4); euclidean and cosine reported as robustness checks.
- Ceiling for condition c: mean over seed pairs of RSA(rep_c,s, rep_c,s'). Cross score RSA(q,c) = mean over all seed
  pairs (q seed × MLP seed). RSA_norm = RSA / sqrt(ceil_q · ceil_mlp).
- Hierarchical bootstrap: resample seeds (with replacement) per condition, then resample probe indices (with
  replacement, duplicates' self-pairs dropped); recompute everything; 2000 resamples; percentile CIs.
- H1: per bootstrap sample compute Spearman(k, RSA_norm_k) and fit 2-segment piecewise-linear (breakpoint grid over
  k ∈ [0.5, L−0.5] step 0.1, least squares, second segment slope free). Report CI of correlation and breakpoint.
- H2: same breakpoint fit on eval return and on −log grad-var-at-init; overlap of CIs; grad-var slope after
  breakpoint < 0.
- H3: pairwise discrepancy D_ij = |r(RDM_q)_ij − r(RDM_mlp)_ij| with r = rank normalised to [0,1]. Pair features:
  Δ = |x̃_i − x̃_j| standardised by probe std; coupled share s = ‖Δ_coupled‖² / ‖Δ‖². Coupled pairs s ≥ 0.75,
  uncoupled s ≤ 0.25; magnitude matched by deciles of ‖Δ‖. Statistic T = mean_coupled(D_sep − D_dense) −
  mean_uncoupled(D_sep − D_dense); null by permuting coupled/uncoupled labels within magnitude deciles (10k).
- Disagreement rule (7.5) evaluated on Spearman(k, CKA_norm) vs Spearman(k, RSA_norm).

## §9 E1 validation (Req 5)
`validation_toy_task.py`: target f(x) = [sin(x1·x2), x1², cos(x3)+x4] on x ∈ U[−1,1]^4. Train two different MLPs
(widths 32 vs 128, different seeds, different depth) to regress f; compare hidden reps → RSA high (≥0.7).
Network trained on unrelated target g → RSA vs f-net low (≤ 0.4 and below the f-f score by ≥0.3).
Invariance: RSA(X, monotone_warp_of_distances) = 1 via RDM-level check; CKA(X, sXQ) = 1. Planted coupling:
rep A = [x1·x2, x3] vs rep B = [x1, x2, x3] — partitioned discrepancy concentrates on pairs differing in (x1,x2) jointly.
Also a quantum-specific check: two PQC instances with identical params but permuted qubit labelling (and permuted
inputs) give RSA = 1.

## §10 Init gradient variance (Req 6.6)
For each (task, k, topology): 200 random inits (θ ~ U[0, 2π], λ = 1), fixed batch of 64 probe states, cost =
mean ⟨Z_0⟩; record Var over inits of ∂cost/∂θ for every parameter, report mean over params (McClean-style).

## §11 Sweep & compute
Grid E2: conditions {q-sep-k0, q-chain-k1..4, q-all-k1..4, mlp, mlp-pm} × seeds 0–4, + seeds 5–9 for
{q-sep-k0, q-chain-k4, mlp}. 70 runs. Process pool with `torch.set_num_threads(1)` per worker, 16 workers.
Ablation: `zzz` observables for the 9 quantum conditions × 5 seeds.

## §12 Gates
G1 (pytest E1 tests) checked by `reproduce.sh` before sweeps. G2 and G3 computed and printed by `analyze.py`
and written to `results/gates.md`; failures change the headline per council rules, never silently.

## §13 Decisions log
See `docs/council/`. Deviation: probe set = pooled visited states of all trained runs (spec §5.3) instead of the
chairman's 50/50 MLP+uniform mix.

## §14 Amendments (dated, with reasons)
- 2026-10-04, E1 threshold: the first E1 run gave RSA(f-net, f-net) = 0.74 and RSA(f-net, g-net) = 0.44, failing the
  pre-written "unrelated ≤ 0.4" criterion. Cause: all networks see the same 4-d inputs, so hidden RDMs inherit input
  geometry (input-geometry floor; RSA(f-net, raw inputs) is reported alongside). The criterion was amended to
  "mean over 3 replicates: same ≥ 0.6, same − diff ≥ 0.2, same > diff in every replicate". Consequence for the
  real experiments: absolute RSA values are uninterpretable without the random-init and input-geometry baselines,
  which are therefore reported for every condition. CKA discriminated less (0.67 vs 0.53) on the same nets.
- 2026-10-04, ordering: H3 partitioned-analysis core (tasks 9) was implemented together with E1 (task 5) because
  the planted-coupling check needs it.
- 2026-10-04, pilot 1 (300k steps, lr actor 2.5e-3 / quantum 1e-2 / critic 1e-3): G2 would fail (eval >= 475 in only 1/3 seeds for q-sep-k0 and q-chain-k4). Every family, MLP included, reached ~500 and then collapsed, so the problem is optimisation instability, not capacity. The one permitted shared adjustment (tasks 6.2): lr actor 5e-4, quantum 5e-3, critic 5e-4, CartPole steps 500k. Frozen after pilot 2 whatever the outcome. Pilot-1 data archived in runs_archive/. Pilot-1 ceilings: RSA q-sep 0.31, q-chain-k4 0.57, MLP 0.39 (G3 provisionally passes).
- 2026-10-04, pilot 2 aborted (actor 5e-4 / quantum 5e-3 / critic 5e-4, 500k): the MLP still collapsed after hitting ~500 even with KL ~2e-4, so the problem was not the policy step size. MLP-only diagnostic, 3 seeds x 4 variants (scratch, not reported as results): CleanRL settings, no entropy and no LR annealing all still collapsed; critic LR 2.5e-3 alone gave eval 500 in 3/3 seeds and flat 499 curves. Diagnosis: the unnormalised-return critic lags behind the policy. The critic is identical across families, so this is a shared fix. FINAL frozen defaults: lr actor 5e-4, quantum 5e-3, critic 2.5e-3, CartPole 500k steps. No further tuning, whatever pilot 3 shows.
