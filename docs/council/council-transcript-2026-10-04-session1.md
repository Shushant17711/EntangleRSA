# Council transcript — Session 1 (2026-10-04): EntangleRSA design decisions

Files: framed-question-1.md (question), s1-anonymized-responses.md (advisors; A=Executor B=Outsider C=Contrarian D=Expansionist E=First Principles),
s1-peer-reviews.md (reviews), s1-chairman-facts.md (verified facts given to the chairman).

## Chairman verdict (summary of binding decisions)
- D1: Keep H1–H3 pre-registered verbatim; 2603.10289 cited as prior against H1. Primary metric Spearman-RSA; linear CKA always reported.
  Noise-ceiling normalisation RSA_norm = RSA(q,c)/sqrt(ceil_q*ceil_c) (ceil = mean pairwise between-seed RSA within condition).
  Pre-registered disagreement rule: if CKA and RSA depth-trends have opposite sign with both 95% CIs excluding 0 → headline "metric-dependent", interpreted via invariance diagnostics.
- D2: PennyLane default.qubit + torch for all runs; custom torch statevector sim kept as pytest oracle (20 random circuits/topology, 1e-5 outputs, 1e-4 grads). Revisit only if a CartPole run > 20 min.
- D3: L=4 re-uploading layers; CZ entanglers after first k layers, k=0..4; topologies chain & all-to-all; 9 conditions (k=0 shared). No IsingZZ.
- D4: Primary rep = <Z_i>; ablation = <Z_i> + all <Z_iZ_j>. Unit test: separable <Z_i> has zero Jacobian wrt x_j (j≠i).
- D5: PPO everywhere (Gaussian head for Pendulum). Fallback: if k=0 and k=4 quantum agents don't reach mean ≥ −400 in 300k steps in 3/5 seeds, Pendulum → H3 probe analysis only, Acrobot (6 qubits) becomes second sweep task.
- D6: 5 seeds per condition; 10 seeds for k=0, k=4 chain, primary MLP (noise ceilings).
- D7: Primary MLP obs→64→4(tanh)→head (width = n_qubits, bounded range); secondary param-matched tanh MLP.
  Invariance diagnostics on every comparison: raw / z-scored / PCA-whitened / rank-transformed reps, both metrics; random-init RSA/CKA per depth (capacity baseline).
- D8: Hierarchical bootstrap (seeds, then probes), 2000 resamples. H1: CI of Spearman(k, RSA_norm) + 2-segment fit with breakpoint CI. H2: breakpoint CI overlap (RSA vs return) + log grad-variance (200 inits/depth) falling beyond breakpoint. H3: 10k-permutation test of separable-minus-MLP dissimilarity, coupled vs matched uncoupled pairs.
- Gates: G1 E1 passes (planted structure, monotone-warp invariance for RSA, rotation invariance for CKA, planted coupling detected). G2 quantum PPO solves CartPole (≥475) 3/5 seeds at k=0 and k=4. G3 quantum seed-to-seed RSA ceiling ≥ 0.2, else reliability becomes headline.
- First thing: E1 + 3-seed CartPole pilot (k=0, k=4 chain, MLP) measuring ceilings, RSA, CKA.

## Orchestrator amendment (recorded deviation)
- Chairman proposed a probe set of 50% MLP-rollout + 50% uniform grid. The spec (§5.3) explicitly requires visited states for the primary probe set.
  Decision: primary probe set = states pooled from rollouts of ALL trained policies (every condition × seed, equal share, so no single policy defines the probe distribution), n=500; uniform-random and size variations move to E5.
