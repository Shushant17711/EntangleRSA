# Gates (pre-registered, design §12)

- **G1 E1 measurement validation: PASS** — 15 passed in 19.39s
- **G2 quantum PPO solves CartPole: PASS** — q-sep-k0: 5/5 seeds eval return ≥ 475; q-chain-k4: 5/5 seeds eval return ≥ 475
- **G3 quantum ceiling ≥ 0.2: FAIL** — min quantum seed-to-seed RSA ceiling = 0.191; below 0.2: {'q-all-k3': 0.191}

Consequences of failure: G1 → no experiment results are reported; G2 → RL sweep scope is cut and
the return axis reported with that caveat; G3 → representational reliability becomes the headline result.
