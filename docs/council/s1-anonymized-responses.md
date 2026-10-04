# Session 1 — anonymized advisor responses

**Response A:**
D1: Keep H1–H3 pre-registered, report H1's contrary result honestly. Add CKA as secondary (~20 lines) and the seed-to-seed noise ceiling — cheap insurance against "why didn't you compute CKA like the paper you respond to."
D2: (c) custom sim cross-validated against PennyLane, used only for the correctness check (E1); run the sweep through PennyLane+torch (a). Writing your own autograd-correct simulator under deadline is how week 3 disappears. lightning.gpu is a trap at 4–8 qubits (dispatch overhead).
D3: Fine k=0..L sweep. Compute trivial; bottleneck is PPO wall-clock. Gives a curve instead of 3 points.
D4: Primary = <Z_i> only — makes H3 falsifiable by design. <Z_i>+<Z_iZ_j> as ablation, same code path.
D5: PPO everywhere; fallback NOW: if quantum PPO on Pendulum doesn't converge within 2 days, reduce Pendulum to qualitative H3 confirmation.
D6: 5 seeds. 3 seeds on a noisy quantity is a coin flip.
D7: Both; primary width-matched (n_qubits-wide MLP) — matched representation dimensionality is what makes RSA comparable.
D8: Bootstrap over probe pairs + seeds; permutation test for H3; simple 2-segment saturation fit. No Bayesian changepoint.
Compute: ~110 CartPole runs × 200k steps, 5–15 min/run, 8–10 in parallel → overnight. Real risk is PPO-on-quantum convergence and simulator correctness, not compute.

**Response B:**
(Outsider view.) The spec is written for someone who already believes the conclusion. A prior paper found the opposite of H1 and the plan is the same experiment with a different correlation coefficient + more seeds — that's a robustness check on someone else's finding. If CKA vs RSA give opposite directions on the same geometry, the metric is doing the work, not the physics. Build a synthetic case with known ground truth where you show whether RSA-Spearman and CKA disagree — week-1 work, not an afterthought.
D1: Re-frame explicitly as "resolving a metric-dependent contradiction"; both CKA and RSA; noise ceilings; don't quietly keep H1.
D2: (c) — only if confident you'll finish cross-validation vs PennyLane.
D3: Fine k sweep — 3 points can't distinguish plateau from noisy line.
D4: <Z_i>+<Z_iZ_j> primary; single-qubit as the ablation that sharpens H3.
D5: PPO; budget real time for Pendulum fallback (SAC or smaller continuous task).
D6: 5+ seeds.
D7: Both; width-matched primary — param matching across gates vs weights isn't apples-to-apples.
D8: Piecewise-linear saturation fits with bootstrap CIs over seeds and probes — permutation tests alone don't say where a plateau starts.

**Response C:**
D1 is a kill-switch question, not a framing one. H1 has a same-year better-powered direct contradiction. Add CKA + noise ceilings, but re-frame: "why do CKA and RDM-Spearman disagree about entanglement's effect, and does that disagreement track anything causal (barren plateaus, H3 localization)". Survives H1 being falsified.
D2: (c) custom sim cross-validated vs PennyLane — PennyLane overhead dominates at 4–8 qubits; you need speed for seeds × depths × tasks × observable sets.
D3: Fine k=0..L, parameter-free entanglers, identical param counts. 3 conditions can't support a saturation curve.
D4: Primary <Z_i> only — otherwise H3 stops being a clean test.
D5: PPO everywhere; pre-register fallback in writing now, else an algorithm swap looks like p-hacking.
D6: 5 seeds minimum.
D7: Both; width-matched primary — param matching lets the classical net brute-force similarity.
D8: Bootstrap over seeds + permutation test for plateau; piecewise fits for visualization only.
Main risk: spending weeks 2–6 confirming H1 false, then scrambling in week 7. Do the reframe now, structurally.

**Response D:**
D1: Keep H1–H3 pre-registered AND add CKA + noise ceilings; treat 2603.10289 as leverage: "return advantage does NOT imply convergence toward classical geometry, and metric choice flips the conclusion" — methods paper + result paper. Lead with it.
D2: (c) custom sim cross-validated vs PennyLane — reusable asset, fast enough to push seed count up.
D3: Fine k=0..L, both topologies, parameter-matched — dose-response curve.
D4: Both <Z_i> and <Z_iZ_j> co-primary; the contrast is itself a finding.
D5: PPO everywhere; add Acrobot if time allows.
D6: 10 seeds to match 2603.10289.
D7: Both; width-matched foregrounded.
D8: Bootstrap + permutation, pre-registered.

**Response E:**
The council is asking the wrong question. The measurement apparatus (RDM-Spearman) was chosen before the claim. 2603.10289 showed another reasonable apparatus (CKA) gives the opposite answer — if two legitimate measures disagree in direction, "RSA(quantum, classical)" isn't a fact about the circuit. Zeroth decision: state the mechanism producing H1 in one causal sentence, independent of the statistic.
D1: Re-frame: "similarity metrics disagree about entanglement's effect — here's why, and here's the invariant that survives metric choice." CKA and RSA co-primary.
D2: (c) — you must own the measurement code.
D3: Fine k sweep, parameter-matched — after D1 is settled.
D4: Both <Z_i> and <Z_iZ_j> co-primary.
D5: PPO everywhere; pre-declare the Pendulum fallback (evolution strategies) now.
D6: 5 seeds minimum.
D7: Both; lead with dimensionality-matched.
D8: Doesn't matter yet — a test on an unstable construct launders noise.
