"""Build price-grade event feature summaries for all 18 historical events.

For Tier A events: uses known empirical values from the benchmark dataset.
For Tier B/C events: uses synthetic estimates derived from max_depeg_bps_est in YAML.

Writes: results/paper_addon/table_17_historical_price_grade_summary.csv

IMPORTANT: All Tier B/C rows are flagged is_synthetic=True.
These estimates MUST use "est." notation in the paper.
Do NOT cite specific percentages from synthetic rows in tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv

import yaml

from stressbench.history.price_grade_features import (
    EventPriceGradeSummary,
    build_all_summaries,
)

YAML_PATH = ROOT / "configs" / "event_windows_historical.yaml"
OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "table_17_historical_price_grade_summary.csv"

FIELDNAMES = [
    "event_id",
    "n_minutes",
    "max_basis_bps",
    "mean_basis_bps",
    "pct_above_10bps",
    "pct_above_50bps",
    "pct_above_100bps",
    "depeg_episode_count",
    "duration_hours",
    "data_sources_used",
    "is_synthetic",
    "claim_level",
    "notes",
]


def main() -> None:
    with open(YAML_PATH, encoding="utf-8") as f:
        events_cfg = yaml.safe_load(f)["events"]

    print(f"Loaded {len(events_cfg)} events from {YAML_PATH.name}")
    summaries: list[EventPriceGradeSummary] = build_all_summaries(events_cfg)

    rows = [s.to_dict() for s in summaries]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    tier_a = sum(1 for s in summaries if s.claim_level == "execution_grade")
    synthetic = sum(1 for s in summaries if s.is_synthetic)
    print(f"Wrote {len(rows)} event summaries → {OUT_FILE}")
    print(f"  Tier A (execution-grade): {tier_a}")
    print(f"  Synthetic (est. only):    {synthetic}")


if __name__ == "__main__":
    main()
