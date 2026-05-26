#!/opt/anaconda3/bin/python
"""
make_stress_summary_table.py
----------------------------
Build the compact 6-row stress-event summary table for the paper.
Sources table_21_stress_case_metrics.csv for Tier-A actual metrics;
uses hardcoded estimates for the Tier-B/C context rows (cross-checked
against event_windows_historical.yaml max_depeg_bps_est values).

Outputs:
  results/paper_addon/table_22_stress_summary_for_paper.csv
"""

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(REPO_ROOT, "results", "paper_addon", "table_21_stress_case_metrics.csv")
OUT_DIR = os.path.join(REPO_ROOT, "results", "paper_addon")
OUT_PATH = os.path.join(OUT_DIR, "table_22_stress_summary_for_paper.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    full = pd.read_csv(IN_PATH)

    # Select the 6 paper rows and the columns needed for the paper table
    selected_ids = [
        "terra_ust_2022",
        "usdc_svb_2023",
        "usdc_svb_recovery_2023",
        "usdt_curve_2023",
        "ftx_collapse_2022",
        "busd_regulatory_2023",
    ]

    rows = []
    for eid in selected_ids:
        r = full[full["event_id"] == eid]
        if r.empty:
            print(f"WARNING: {eid} not found in table_21 — skipping")
            continue
        r = r.iloc[0]
        rows.append(r)

    df = pd.DataFrame(rows)

    # Rename and select columns for the paper-facing table
    out = pd.DataFrame({
        "event_id": df["event_id"],
        "event_name": df["event_name"],
        "mechanism_class": df["mechanism_class"],
        "data_tier": df["data_tier"],
        "role_in_paper": df["role_in_paper"],
        "n_minutes": df["n_minutes"],
        "max_abs_basis_bps": df["max_abs_basis_bps"],
        "pct_gt_10bps": df["pct_gt_10bps"],
        "exec_label_available": df["exec_label_available"],
        "exec_positive_rate_5m_q10k": df["exec_positive_rate_5m_q10k"],
        "claim_allowed": df["claim_allowed"],
    })

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows → {OUT_PATH}")
    print()
    print(out[["event_id", "data_tier", "role_in_paper", "n_minutes",
               "pct_gt_10bps", "exec_label_available", "exec_positive_rate_5m_q10k"]].to_string(index=False))


if __name__ == "__main__":
    main()
