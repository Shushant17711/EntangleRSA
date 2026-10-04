# Council session 1 — EntangleRSA design decisions

## Project
EntangleRSA (8-week research project, single person, laptop with RTX 4060 8GB, 20 cores, 23GB RAM).
Question: does adding entangling gates to a parameterized-quantum-circuit (PQC) RL policy change its
internal representational geometry (measured with Representational Similarity Analysis: RDM of
pre-measurement expectation values vs RDM of a classical MLP's penultimate layer, compared by Spearman),
not just its return? Pre-stated hypotheses:
- H1: RSA(quantum, classical) increases with entangling depth up to saturation, then plateaus.
- H2: RSA saturation depth coincides with return saturation depth; beyond it gradient variance falls (barren-plateau proxy). Central figure: RSA, return, grad-variance vs entangling depth.
- H3: Separable circuits show higher RDM dissimilarity to the MLP specifically on probe pairs differing along a known coupled dimension (CartPole pole-angle x cart-velocity; Pendulum angle x angular velocity).
Experiments: E1 synthetic RSA sanity check; E2 CartPole sweep 3 seeds; E3 Pendulum; E4 localized (H3); E5 probe-set ablation. Conditions in spec: separable / linear-chain (CZ nearest neighbour) / dense (CZ or IsingZZ all-to-all), parameter counts matched; PPO preferred; MLP baseline with matched budget.

## NEW evidence from the week-1 literature check
arXiv:2603.10289 (Wang, Hymas, Quach 2026, "Quantum entanglement provides a competitive advantage in adversarial games"):
8-qubit data-reuploading PQC (U3 gates, trainable input weights/biases, 48 params/layer) as PPO feature extractor on Pong (8-dim obs),
separable vs CZ vs IsingZZ, 1–6 layers, 10 seeds. Entangled >> separable on return; returns peak at shallow depth.
They ran LINEAR CKA (not RDM/Spearman RSA): classical MLPs highly similar to each other; quantum backbones have LOW intra-group
similarity across seeds (esp. CZ); ALL quantum reps share little similarity with classical; separable is MORE similar to classical
than entangled — i.e. the opposite direction to H1. They did NOT: sweep RSA against depth, measure gradient variance, use noise
ceilings / seed-to-seed reliability, localize dissimilarity (H3), use single-agent control tasks (CartPole/Pendulum), or RDM-Spearman.
So the "gap" claimed in the spec is partly closed; H1 has prior contrary evidence.

## Decisions to make (give a concrete pick for each)
D1. Framing/novelty given 2603.10289: keep H1–H3 as pre-registered predictions (and report contrary result honestly), or re-frame? Should we add CKA as a secondary metric for direct comparability and a noise ceiling (quantum seed-to-seed RSA, classical seed-to-seed RSA) to normalise RSA?
D2. Simulator stack: (a) PennyLane default.qubit + torch interface (batched), (b) PennyLane lightning.gpu, (c) custom batched PyTorch statevector simulator (4–8 qubits, 2^n ≤ 256 amplitudes, exact expectation values, autograd), cross-validated against PennyLane in unit tests. Python 3.14 is system default; uv available.
D3. Definition of the "entangling depth" axis for H1/H2: e.g. fixed L variational layers (data re-uploading) with entanglers inserted after the first k layers, k = 0..L, for topologies chain and all-to-all with parameter-free CZ (so parameter counts are EXACTLY identical across all conditions), vs. the spec's 3 coarse conditions only, vs. varying L.
D4. Encoding & observables: qubits = obs dim (CartPole 4; Pendulum 3 [cos,sin,thetadot]; Acrobot 6). Data re-uploading with trainable input scaling (Jerbi 2021). Representation = single-qubit <Z_i> only (then a separable circuit's <Z_i> is provably a function of x_i alone — makes H3 sharp) vs. <Z_i> + <Z_iZ_j> (separable circuits then produce products f(x_i)g(x_j), which DO encode joint info). Which is primary, which ablation?
D5. RL algorithm: PPO for all tasks (discrete CartPole/Acrobot, Gaussian head for Pendulum)? Quantum PPO on Pendulum is notoriously hard; fallback?
D6. Seeds/compute: 3 seeds (spec) vs 5+ given the 2603.10289 finding that quantum reps vary a lot across seeds and our memory that RL results have high re-roll variance.
D7. Classical baseline matching criterion: primary = MLP with penultimate width = n_qubits (same representation dimensionality) and roughly matched params; alternative = param-count-matched; report both.
D8. Statistical test for H1/H2/H3: what exactly (e.g. bootstrap over probe states + seeds, permutation tests, piecewise-linear saturation fits)?

## Stakes
8 weeks of work; the paper's credibility hinges on (i) not being scooped/redundant with 2603.10289, (ii) a pipeline that is demonstrably correct (E1), (iii) results not being artifacts of seeds/probes/matching. Wrong early choices (stack, depth axis, observable set) are expensive to undo.
