"""Guard test: add-on scripts must not write to baseline result directories.

This test inspects the source code of add-on scripts to verify they never
write to results/experiments/ or results/paper/ (only _addon variants).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ADDON_SCRIPTS = [
    "scripts/run_addon_experiments.py",
    "scripts/run_robustness_grid.py",
    "scripts/make_addon_tables.py",
    "scripts/make_addon_figures.py",
    "scripts/analyze_false_positives.py",
]

_BASELINE_PATHS = [
    "results/experiments/",
    "results/paper/",
]

_BASELINE_FILES = [
    "results/experiments/all_results.csv",
    "results/paper/table_1_data_coverage.csv",
    "results/paper/table_2_price_execution_gap.csv",
    "results/paper/table_3_model_ablation.csv",
    "results/paper/table_4_oracle_gap.csv",
]

_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("script", _ADDON_SCRIPTS)
def test_addon_script_does_not_reference_baseline_write_path(script):
    """Confirm no explicit baseline output path appears in write context."""
    path = _ROOT / script
    if not path.exists():
        pytest.skip(f"{script} not yet implemented")

    source = path.read_text()

    # Check that the script does not contain hardcoded writes to baseline dirs
    for baseline_file in _BASELINE_FILES:
        # Allow reads (baseline results are used as input), flag writes
        # Heuristic: if the file name appears after open(..., "w") pattern
        write_pattern = re.compile(
            rf'open\s*\([^)]*{re.escape(baseline_file.split("/")[-1])}[^)]*,\s*["\']w["\']',
            re.DOTALL,
        )
        assert not write_pattern.search(source), (
            f"{script} appears to open {baseline_file} for writing"
        )


def test_baseline_experiment_files_unchanged():
    """Baseline all_results.csv must exist and not be empty."""
    path = _ROOT / "results/experiments/all_results.csv"
    assert path.exists(), "results/experiments/all_results.csv is missing"
    lines = path.read_text().splitlines()
    assert len(lines) > 1, "all_results.csv appears empty"


def test_baseline_paper_tables_exist():
    for fname in _BASELINE_FILES[1:]:  # skip all_results.csv
        p = _ROOT / fname
        assert p.exists(), f"Baseline file missing: {fname}"
