# Implementation Plan — EntangleRSA

Strategy: freeze contracts (config, env specs, circuit), prove measurement correctness first (E1 + simulator
oracle), then get one quantum and one classical agent training end-to-end (pilot), then scale to sweeps and
analysis. "Done" = `scripts/reproduce.sh` regenerates `results/` (the three-curve figure, the H3 figure,
main table, gates) from scratch, all tests green, and README/LIMITATIONS/paper reflect the actual numbers.

## Out-of-band prerequisites
- Literature check done 2026-10-04: arXiv:2603.10289 already ran linear CKA (not RSA) quantum-vs-classical on Pong;
  no depth sweep, no grad variance, no localisation, no noise ceilings. Recorded in `docs/council/`.
- Council decisions binding (session 1).

## Task list

- [x] 1. Implement RSA core primitives
  - `entangle_rsa/rsa/core.py`: rdm (cosine/correlation/euclidean), upper_tri, spearman, rsa_score, cka_linear
  - `tests/test_rsa_core.py`: rotation/padding invariance, scipy agreement, independence ≈ 0, CKA properties
  - _Requirements: 4.1_

- [x] 2. Freeze shared contracts
- [x] 2.1 Add run configuration and condition naming
  - `entangle_rsa/config.py`: RunConfig dataclass (task, family, topology, k, obs, seed, steps, PPO hparams), `cond_name()`, `run_dir()`, JSON round-trip
  - tests: k=0 maps to `q-sep-k0` for both topologies; round-trip equality
  - _Requirements: 1.2, 3.3_
- [x] 2.2 Add environment wrappers and task specs
  - `entangle_rsa/envs/wrappers.py`: TaskSpec(name, obs_dim, act, scale, coupled_dims, uncoupled_dims), `make_env`, `make_vec_env`, `scale_obs`
  - tests: scaled CartPole obs within ~[−2,2] over random rollouts; Pendulum angle helper
  - _Requirements: 3.2, 6.3_

- [x] 3. Build the quantum circuit with a verified simulator
- [x] 3.1 Implement the torch statevector oracle
  - `policies/statevector.py`: same layer structure (RY,RZ | CZ×topology if l<k | RX(λx)), Z and ZZ expectations
  - _Requirements: 1.5_
- [x] 3.2 Implement the PennyLane circuit and QuantumActor
  - `policies/circuits.py` (entangler pairs, observables, qnode builder with broadcasting); `policies/quantum_policy.py` (params θ, λ; `rep(x)`, `forward`)
  - tests: PennyLane vs oracle on 20 random circuits × topology (outputs ≤1e-5, grads ≤1e-4); param count identical across k/topology; separable Jacobian ∂⟨Z_i⟩/∂x_j = 0 (j≠i) and non-zero for entangled
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 4. Implement classical actors, critic and heads
  - `policies/classical_policy.py`: MLPActor (primary width = n, tanh rep), `param_matched_hidden()`, Critic; `policies/heads.py`: Categorical/Gaussian heads
  - tests: rep shape/range, param-matched within ±10% of quantum actor params
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Write and pass the E1 validation suite
  - `rsa/transforms.py` (zscore, pca_whiten, rank); `rsa/validation_toy_task.py` (toy regression, two-net RSA, unrelated target, invariances, planted coupling, qubit-permutation check)
  - `tests/test_validation_e1.py` asserting thresholds from design §9 — this is gate G1
  - _Requirements: 4.5, 5.1, 5.2_

- [ ] 6. Implement PPO training end-to-end
- [x] 6.1 Implement the PPO trainer with grad-variance logging
  - `rl/ppo.py`: rollout, GAE, clipped loss, param groups (quantum vs classical LR), per-minibatch grad collection → grad_var
  - `train.py` CLI writes config.json, metrics.csv, model.pt, eval.json, `done`
  - test: 2-update smoke run for MLP and quantum on CartPole produces all artifacts
  - _Requirements: 3.1, 3.3, 3.4_
- [ ] 6.2 Run the CartPole pilot and freeze hyperparameters
  - pilot: q-sep-k0, q-chain-k4, mlp × 3 seeds; adjust shared LR/steps once if G2 fails, then freeze in config defaults
  - record pilot results in `results/pilot.md`
  - _Requirements: 3.1, 8.3_

- [ ] 7. Build probe collection and representation extraction
  - `rsa/probes.py` (pooled trajectory probes with equal per-run share; uniform probes); `rsa/extract.py` (load run → rep matrix for given probes; random-init reps)
  - tests: deterministic probes for fixed seed; extract shape (n_probes, d) for quantum and MLP
  - _Requirements: 4.2, 4.3, 4.6_

- [x] 8. Implement the statistics layer
  - `rsa/stats.py`: ceilings, cross-RSA/CKA matrices, RSA_norm, hierarchical bootstrap, 2-segment piecewise fit, disagreement rule
  - tests: piecewise fit recovers a planted breakpoint; bootstrap CI covers truth on synthetic data; ceiling = 1 for identical seeds
  - _Requirements: 4.4, 7.1, 7.2, 7.3, 7.5_

- [x] 9. Implement the H3 partitioned analysis
  - `rsa/partitioned_analysis.py`: pair features, coupled share, magnitude-matched partitions, discrepancy, permutation test
  - test: planted-coupling synthetic reps give significant T; null reps don't
  - _Requirements: 6.3, 7.4_

- [x] 10. Implement init gradient variance
  - `analysis/gradvar_init.py`: 200 inits per (task, k, topology), mean param variance of ∂⟨Z_0⟩/∂θ, CSV output
  - _Requirements: 6.6, 7.3_

- [ ] 11. Run the sweeps
- [ ] 11.1 Add the sweep runner and run E2 CartPole
  - `sweep.py`: grid → process pool (1 thread/worker), skip `done` runs, log failures; E2 grid incl. extra ceiling seeds
  - _Requirements: 3.5, 6.1_
- [ ] 11.2 Run E3 Pendulum with pre-registered fallback check
  - check fallback criterion programmatically; if triggered, run Acrobot sweep and restrict Pendulum to H3
  - _Requirements: 6.2_
- [ ] 11.3 Run the observable ablation (zzz) on CartPole
  - _Requirements: 6.5_

- [ ] 12. Build the analysis and figures
  - `analysis/aggregate.py` + `analyze.py`: probes, reps, RSA/CKA + diagnostics, bootstrap, H1/H2/H3 verdicts, E5 ablation, gates → `results/rsa_return_gradvar_curve.png`, `results/partitioned_dissimilarity.png`, `results/main_table.md`, `results/gates.md`
  - _Requirements: 4.5, 6.4, 7.2, 7.3, 7.4, 7.5, 8.1, 8.3_

- [ ] 13. Write reproduce script, README, LIMITATIONS, paper
  - `scripts/reproduce.sh` (G1 tests → sweeps → analyze); README with matching criteria and results; LIMITATIONS.md; `paper/main.tex` filled from results
  - _Requirements: 2.3, 5.3, 8.2_

## Milestones
- After 5: measurement validated (G1).
- After 6.2: G2 known; pilot ceilings give an early read on G3.
- After 12: the paper's central figure exists.

## Coverage
1.1–1.5 → 3.x · 2.x → 4, 13 · 3.1–3.4 → 2, 6 · 3.5 → 11.1 · 4.1 → 1 · 4.2–4.3 → 7 · 4.4 → 8 · 4.5 → 5, 12 · 4.6 → 7 ·
5.1–5.2 → 5 · 5.3 → 13 · 6.1 → 11.1 · 6.2 → 11.2 · 6.3 → 2.2, 9 · 6.4 → 12 · 6.5 → 11.3 · 6.6 → 10 · 7.x → 8, 9, 10, 12 · 8.x → 6.2, 12, 13
