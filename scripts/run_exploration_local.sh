#!/usr/bin/env bash
# One-command local exploration + results extraction.
# Usage: bash scripts/run_exploration_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PYTHONPATH=src "$PY" scripts/explore_executable_transfer.py
PYTHONPATH=src "$PY" scripts/extract_exploration_results.py
echo "Results: results/exploration/SUMMARY.md  +  figure_exploration_summary.png  +  executable_transfer_ledger.csv"
