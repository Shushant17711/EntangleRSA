# Requirements — EntangleRSA

Source: `13-qrl-entanglement-rsa.md` (project brief) + council session 1
(`docs/council/council-transcript-2026-10-04-session1.md`). Hypotheses H1–H3 are pre-registered verbatim
from the brief §4 and are not edited after results come in.

## 1. Quantum policies
- **1.1** PQC policy with data re-uploading (trainable input scaling), qubits = observation dim, L=4 layers.
- **1.2** Entangling depth k ∈ {0..L}: parameter-free CZ entanglers after the first k variational layers; topologies `chain` and `all` (all-to-all). Parameter count identical across all k/topologies.
- **1.3** Representation = pre-head expectation values: primary ⟨Z_i⟩; ablation ⟨Z_i⟩ + ⟨Z_iZ_j⟩ (all i<j).
- **1.4** A separable circuit's ⟨Z_i⟩ depends on x_i only (verified by a Jacobian test).
- **1.5** Simulation via PennyLane default.qubit + torch; an independent torch statevector oracle agrees on outputs (≤1e-5) and gradients (≤1e-4) on 20 random circuits per topology.

## 2. Classical baselines
- **2.1** Primary MLP: obs → 64 (tanh) → n_qubits (tanh) → head (representation width = n_qubits, bounded range).
- **2.2** Secondary MLP: tanh MLP whose parameter count matches the quantum policy's actor parameters (±10%).
- **2.3** Matching criteria stated explicitly in outputs/README.

## 3. RL training
- **3.1** PPO for all policies; categorical head for discrete tasks, Gaussian head for Pendulum. The critic is an identical classical MLP across all conditions.
- **3.2** Tasks: CartPole-v1 (primary), Pendulum-v1, Acrobot-v1 (fallback/stretch) via thin wrappers with fixed, documented observation scaling.
- **3.3** Every run is reproducible from (task, policy config, seed); saves checkpoint, per-update metrics CSV, config JSON.
- **3.4** Log gradient variance of actor parameters across minibatches during training.
- **3.5** Sweep runner executes a grid of runs in parallel processes and skips completed runs.

## 4. RSA pipeline
- **4.1** RDM (cosine / correlation / euclidean), Spearman RSA on upper triangles, linear CKA.
- **4.2** Probe set: states pooled from rollouts of all trained policies (equal share per run), default n=500, fixed seed; alternative uniform-random probes.
- **4.3** Representation extraction for quantum (expectation values) and classical (penultimate activations) over the same probes.
- **4.4** Noise ceilings: mean pairwise between-seed RSA within a condition; RSA_norm = RSA(q,c)/sqrt(ceil_q·ceil_c).
- **4.5** Invariance diagnostics: RSA and CKA on raw / z-scored / PCA-whitened / rank-transformed representations.
- **4.6** Random-initialisation RSA/CKA per depth (capacity/architecture baseline).

## 5. Validation (E1)
- **5.1** Synthetic toy task: two different networks trained to compute the same function score high RSA; unrelated functions score low.
- **5.2** Invariance checks: RSA invariant to monotone warps of distances; CKA invariant to rotation/isotropic scaling; both detect a planted coupling.
- **5.3** Gate G1: real experiments must not run unless E1 tests pass (enforced by the reproduce script).

## 6. Experiments
- **6.1** E2 CartPole sweep: 9 quantum conditions × 5 seeds + 2 MLPs × 5 seeds; extra seeds to 10 for k=0, k=4 chain, primary MLP.
- **6.2** E3 Pendulum sweep (3 qubits) with pre-registered fallback: if k=0 and k=4 quantum agents don't reach mean return ≥ −400 within 300k steps in 3/5 seeds → Pendulum restricted to H3 analysis and Acrobot becomes the second sweep task.
- **6.3** E4 localized dissimilarity (H3) on CartPole (coupled dims θ, ẋ) and Pendulum (coupled: angle, angular velocity).
- **6.4** E5 probe ablation: probe size {100, 200, 500, 1000} × sampling {trajectory, uniform}.
- **6.5** Observable ablation (⟨Z⟩+⟨ZZ⟩) on CartPole.
- **6.6** Gradient variance at init: 200 random inits per (k, topology) — barren-plateau proxy.

## 7. Statistics
- **7.1** Hierarchical bootstrap (seeds, then probes), 2000 resamples, RDMs/ceilings recomputed per resample.
- **7.2** H1: 95% CI of Spearman(k, RSA_norm) + 2-segment piecewise fit with bootstrap breakpoint CI.
- **7.3** H2: breakpoint CIs of RSA and return overlap, and log grad-variance decreases beyond the breakpoint.
- **7.4** H3: permutation test (10k) on separable-minus-MLP discrepancy for coupled vs magnitude-matched uncoupled probe pairs.
- **7.5** Metric-disagreement rule: opposite-sign CKA vs RSA trends with both CIs excluding 0 → "metric-dependent" verdict.

## 8. Outputs
- **8.1** `results/rsa_return_gradvar_curve.png` (three curves vs entangling depth), `results/partitioned_dissimilarity.png`, `results/main_table.md`.
- **8.2** README (setup, reproduce), LIMITATIONS.md, `scripts/reproduce.sh`, `paper/main.tex` populated from results.
- **8.3** Gate G2: quantum PPO solves CartPole (≥475) in ≥3/5 seeds at k=0 and k=4 — reported; G3: quantum ceiling ≥0.2 — reported, with reliability becoming the headline if failed.
