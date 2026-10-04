# EntangleRSA: What Does Entanglement Actually Buy a Quantum RL Policy?

> Measure whether entangling gates change a quantum policy's internal representational geometry
> — not just its accuracy — using Representational Similarity Analysis, the standard tool
> computational neuroscience uses to compare what two different systems have actually learned.

---

## 1. The question this answers that accuracy alone can't

The usual argument for entanglement in a parameterized quantum circuit used as an RL policy is
a parameter-counting one: entangling gates let the circuit encode joint information between
input variables (the PDF's own example — paddle motion and ball trajectory in Pong) that a
separable circuit structurally cannot represent. That's a claim about *representation*, but it's
almost always tested by comparing final task performance, which is a claim about *outcome*. Two
policies can reach the same return through completely different internal representations, or
reach different returns while representing the task almost identically. Accuracy alone can't
distinguish these cases. Representational Similarity Analysis can.

## 2. What RSA is, briefly

For a fixed set of probe inputs (here, states sampled from training trajectories), compute a
**Representational Dissimilarity Matrix (RDM)**: the pairwise dissimilarity (1 − correlation, or
cosine distance) between the system's internal representation of each pair of probe states. Two
systems — a quantum circuit and a classical network — are compared not representation-to-
representation directly (their dimensionalities differ) but **RDM-to-RDM**, via a Spearman
correlation between the two dissimilarity matrices. This sidesteps the dimensionality mismatch
entirely and is exactly why RSA is the standard tool for comparing brains to neural networks in
computational neuroscience — the comparison problem is structurally identical here.

## 3. The gap

Quantum representation-learning claims are typically supported by accuracy comparisons or by
expressivity bounds (circuit capacity arguments). **A systematic RSA comparison between quantum
and classical policy representations, swept across entangling depth, in a reinforcement
learning setting, does not appear in the literature** — check this directly before committing
(a focused search for "representational similarity analysis" combined with "quantum neural
network" or "quantum reinforcement learning" is the first thing to do in week 1; supervised
QML work may have touched this, RL settings less likely have).

## 4. Claims you are testing

> **H1.** RSA correlation between the quantum policy's pre-measurement representation and a
> matched classical MLP's penultimate-layer representation increases with entangling depth up
> to a saturation point, then plateaus.

> **H2.** The RSA saturation point coincides with the task-performance saturation point —
> additional entanglement beyond what's needed to match the classical representational
> structure buys no further return, but does increase trainability risk (falling gradient
> variance, a barren-plateau proxy). This is the paper's central figure: three curves — RSA
> score, episodic return, gradient variance — all plotted against entangling depth on one axis.

> **H3.** For tasks with a known coupled degree of freedom (Pendulum's angle–angular-velocity
> coupling; CartPole's pole-angle–cart-velocity coupling), separable (non-entangled) circuits
> show systematically higher RDM dissimilarity to the classical MLP specifically on probe pairs
> that differ along the coupled dimension — i.e., the representational gap is localized to
> exactly the structure entanglement is supposed to capture, not diffuse across the whole state
> space.

H3 is the sharpest and most falsifiable claim — it predicts *where* in representation space the
gap should appear, not just that a gap exists.

## 5. Method

### 5.1 Policies to compare

Build a family of PQC-based policies (PPO or REINFORCE — PPO preferred for stability) with
systematically varied entangling structure:

| Condition | Entangling gates |
|---|---|
| Separable | None — pure single-qubit rotation layers |
| Linear-chain | CZ or CNOT along a 1D nearest-neighbor chain |
| Dense | IsingZZ or CZ, all-to-all connectivity |

Match parameter count as closely as possible across conditions by adjusting the number of
rotation layers, so differences in performance and representation can't be attributed to raw
parameter budget alone.

**Classical baseline:** an MLP with a matched (not identical — quantum and classical parameter
counts aren't directly comparable, state this explicitly) parameter budget, trained on the same
task with the same algorithm.

### 5.2 Tasks

Start with **CartPole-v1** — cheap, well-understood, has one clean coupled pair of state
variables. Repeat on **Pendulum-v1** (continuous control, different coupling structure) once
the pipeline is validated. Consider **Acrobot-v1** as a stretch goal — more genuinely
multi-variable coupling, harder to solve, useful third data point.

### 5.3 Extracting the representation

The "representation" you can actually extract from a quantum policy is the vector of measured
expectation values (over a fixed observable set) that feeds the classical policy head — you
cannot access the full quantum state directly, and treating anything else as "the
representation" would be a category error. For the classical MLP, use the penultimate layer's
activations. Both are extracted over the same fixed probe set: a batch of states sampled from
saved trajectories (not randomly generated states — use states the policy actually visits, or
the RDM reflects an irrelevant part of state space).

### 5.4 RSA pipeline

```
probe_states = sample_from_saved_trajectories(policy, n=200)
quantum_reps = [get_expectation_values(quantum_policy, s) for s in probe_states]
classical_reps = [get_penultimate_activations(classical_policy, s) for s in probe_states]

RDM_q = pairwise_dissimilarity(quantum_reps)      # 200×200, cosine or 1-correlation
RDM_c = pairwise_dissimilarity(classical_reps)

rsa_score = spearman_correlation(upper_triangle(RDM_q), upper_triangle(RDM_c))
```

**Validation checkpoint before any real experiment:** run this pipeline on a synthetic toy task
with a known representational structure you constructed yourself (e.g., two networks trained to
explicitly compute the same simple function) and confirm the RSA score correctly reports high
similarity. If your pipeline can't pass a sanity check with a known answer, don't trust it on
the real comparison.

## 6. Experiments

- **E1 — Pipeline validation.** The synthetic sanity check above.
- **E2 — Entangling depth sweep, CartPole.** Train separable / linear-chain / dense policies,
  3 seeds each. Compute RSA score, episodic return, and gradient variance for each. This is the
  main figure (H1, H2).
- **E3 — Repeat on Pendulum.** Confirms the pattern isn't CartPole-specific.
- **E4 — Localized dissimilarity analysis (H3).** For CartPole and Pendulum, partition probe
  pairs by whether they differ along the known coupled dimension vs. an uncoupled one. Compare
  RDM dissimilarity gap (separable vs. dense) within each partition.
- **E5 — Ablation.** Vary the probe set size and sampling strategy (trajectory-sampled vs.
  uniform-random states) to check H1–H3 aren't artifacts of probe choice.

## 7. Metrics

- RSA (Spearman) correlation, quantum vs. classical RDM
- Episodic return, mean ± std over seeds
- Gradient variance across training (barren-plateau proxy — `Var[∂θ L]` measured periodically)
- Partitioned RDM dissimilarity gap (H3)

## 8. Repository structure

```
entangle-rsa/
├── README.md
├── LIMITATIONS.md
├── policies/
│   ├── quantum_policy.py       # separable / linear-chain / dense variants
│   └── classical_policy.py     # matched-capacity MLP
├── envs/                        # thin wrappers over Gymnasium CartPole/Pendulum/Acrobot
├── rsa/
│   ├── pipeline.py
│   ├── validation_toy_task.py  # E1 sanity check
│   └── partitioned_analysis.py # H3
├── train.py
├── scripts/reproduce.sh
├── results/
│   ├── rsa_return_gradvar_curve.png   # THE figure
│   ├── partitioned_dissimilarity.png
│   └── main_table.md
└── paper/main.tex
```

## 9. Timeline (8 weeks)

| Week | Work |
|---|---|
| 1 | Literature check — confirm the RSA-for-QRL gap. Read Kriegeskorte et al. on RSA methodology. |
| 2 | Build the three policy variants + classical baseline. Get all four training on CartPole. |
| 3 | Build and validate the RSA pipeline (E1). |
| 4 | Full CartPole sweep, 3 seeds (E2). |
| 5 | Pendulum replication (E3). |
| 6 | Localized dissimilarity analysis (E4). |
| 7 | Probe-set ablation (E5). Acrobot stretch goal if time allows. |
| 8 | Write-up. |

## 10. Compute

The cheapest project in this set of five. Shallow circuits (4–8 qubits), thousands of short
episodes — exactly the regime the RTX 4060 handles well via PennyLane's vectorization and CUDA
batching. No density-matrix simulation needed unless you add a noise-robustness extension later.

## 11. Failure modes

- **RSA is sensitive to probe set choice.** Address directly via E5 rather than picking one
  probe set and hoping it's representative.
- **Classical MLP baseline isn't actually capacity-matched.** Parameter-count matching between
  quantum and classical models is inherently approximate — state your matching criterion
  explicitly and report results are robust to at least one alternative matching choice.
- **PPO training instability** is a common noise source in small RL experiments — 3+ seeds and
  reporting variance is mandatory, not optional, given how sensitive RSA scores could be to
  which local optimum a given seed lands in.
- **Null result.** If RSA and performance saturate at completely different entangling depths
  (no H2 relationship), that's a real and reportable finding — entanglement's representational
  and performance benefits would then be shown to be decoupled, which is itself interesting.

## 12. References

- Kriegeskorte, Mur, Bandettini, *Representational Similarity Analysis — Connecting the Branches
  of Systems Neuroscience* (2008) — the foundational RSA methodology paper
- Jerbi, Gyurik, Marshall, Briegel, Dunjko, *Parametrized Quantum Policies for Reinforcement
  Learning* (NeurIPS 2021)
- Skolik, Jerbi, Dunjko, *Quantum Agents in the Gym: A Variational Quantum Algorithm for Deep
  Q-Learning* (Quantum, 2022)
- McClean, Boixo, Smelyanskiy, Babbush, Neven, *Barren Plateaus in Quantum Neural Network
  Training Landscapes* (2018) — for the gradient-variance trainability proxy
- The PDF-cited arXiv:2603.10289 (entanglement competitive advantage claim) — verify its exact
  scope before quoting it, it was not independently checked for this README
