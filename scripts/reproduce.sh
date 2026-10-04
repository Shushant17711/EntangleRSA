#!/usr/bin/env bash
# Regenerate every result from scratch: G1 gate (E1 tests) -> sweeps -> analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORKERS=${WORKERS:-16}

echo "== G1: measurement validation (E1) and simulator oracle =="
$PY -m pytest -q -m "" tests/  # includes slow E1 tests; failure aborts before any experiment

echo "== E2: CartPole sweep =="
$PY sweep.py e2 --workers "$WORKERS"
$PY analyze.py --task cartpole --workers "$WORKERS"

echo "== E3: Pendulum (pre-registered fallback is checked inside) =="
$PY sweep.py e3 --workers "$WORKERS"
$PY -m entangle_rsa.analysis.fallback --task pendulum || {
  echo "Pendulum fallback triggered: running Acrobot as second sweep task";
  $PY sweep.py e3-acrobot --workers "$WORKERS";
  $PY analyze.py --task acrobot --workers "$WORKERS"; }
$PY analyze.py --task pendulum --workers "$WORKERS"

echo "== E6 observable ablation (zzz) and E5 probe ablation =="
$PY sweep.py ablation-zzz --workers "$WORKERS"
$PY -m entangle_rsa.analysis.ablation --task cartpole

echo "== gates =="
$PY -m entangle_rsa.analysis.gates
