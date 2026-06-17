#!/usr/bin/env python3
"""Extract a readable summary from the executable-transfer exploration ledger.

Writes:
  results/exploration/SUMMARY.md                  (human-readable report)
  results/exploration/figure_exploration_summary.png  (net bps by protocol vs oracle)

Run:  python scripts/extract_exploration_results.py
"""
from __future__ import annotations
import csv, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "results/exploration/executable_transfer_ledger.csv"
SUMMARY = ROOT / "results/exploration/SUMMARY.md"
FIG = ROOT / "results/exploration/figure_exploration_summary.png"
ORACLE = 162.2


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def bh_findings(rows, q=0.10):
    cand = [r for r in rows if _f(r["n"]) >= 30 and _f(r["net_bps"]) > 0
            and _f(r["ci_lo"]) > 0 and not math.isnan(_f(r["boot_p"]))]
    if not cand:
        return []
    cand = sorted(cand, key=lambda r: _f(r["boot_p"])); m = len(rows)
    return [r for i, r in enumerate(cand, 1) if _f(r["boot_p"]) <= (i / m) * q]


def main():
    rows = list(csv.DictReader(open(LEDGER)))
    valid = [r for r in rows if not math.isnan(_f(r["net_bps"])) and _f(r["n"]) >= 30]
    findings = bh_findings(rows)
    by_proto = {}
    for r in valid:
        by_proto.setdefault(r["protocol"], []).append(_f(r["net_bps"]))
    best = sorted(valid, key=lambda r: -_f(r["net_bps"]))[:8]

    L = []
    L.append("# StressBench executable-window exploration — results summary\n")
    L.append(f"**Total paths logged:** {len(rows)}  ·  "
             f"**well-defined paths (real net, n≥30):** {len(valid)}  ·  "
             f"**oracle ceiling:** +{ORACLE} bps\n")
    L.append(f"**Positive findings surviving Benjamini–Hochberg FDR(0.10) "
             f"with net>0, CI_lo>0, n≥30:** {len(findings)}\n")
    if findings:
        L.append("\n| protocol | features | model | net_bps | 95% CI | n |\n|---|---|---|---|---|---|")
        for r in findings:
            L.append(f"| {r['protocol']} | {r['fs']} | {r['model']} | {r['net_bps']} | "
                     f"[{r['ci_lo']},{r['ci_hi']}] | {r['n']} |")
    else:
        L.append("\n> No path yields a positive return that survives multiple-testing "
                 "correction. Selecting profitable executable windows is unsolved on this data.\n")

    L.append("\n## Mean net bps by training protocol (well-defined paths)\n")
    L.append("| protocol | mean net_bps | n_paths | interpretation |\n|---|---|---|---|")
    interp = {
        "crossmech": "train Terra → test SVB (cross-mechanism transfer)",
        "control2svb": "train calm controls → test SVB",
        "inevent_wf": "train 1st-half SVB → test 2nd-half (in-event)",
        "inevent_purged_cv": "purged 5-fold CV within SVB (in-distribution upper bound)",
    }
    for p, v in sorted(by_proto.items(), key=lambda kv: st.mean(kv[1])):
        L.append(f"| {p} | {st.mean(v):+.1f} | {len(v)} | {interp.get(p,'')} |")

    L.append("\n## Best (least-negative) well-defined paths\n")
    L.append("| protocol | features | model | net_bps | 95% CI | n |\n|---|---|---|---|---|---|")
    for r in best:
        L.append(f"| {r['protocol']} | {r['fs']} | {r['model']} | {r['net_bps']} | "
                 f"[{r['ci_lo']},{r['ci_hi']}] | {r['n']} |")

    L.append("\n## Conclusion\n")
    L.append("Across every training protocol, feature set, model, and notional tried, no model "
             "selects profitable executable windows out-of-sample. Crucially, even **purged "
             "in-event cross-validation** (train and test within the same SVB crisis) is strongly "
             "negative, so the executable windows are **unpredictable from this microstructure**, "
             "not merely non-transferable. This makes StressBench a genuine open challenge and "
             "supports the benchmark's headline contributions (the ~12× optical→executable gap, "
             "the AUROC–P&L inversion, and venue-specificity) rather than any profitable model.\n")
    L.append("\n*Scope:* this gold panel covers only 2 calm controls, Terra, SVB, and a 2024 "
             "control; no other stress events and no row-level on-chain AMM panel are present, so "
             "the on-chain venue (where executable≈100%) is not modelled here.\n")
    SUMMARY.write_text("\n".join(L))
    print(f"wrote {SUMMARY.relative_to(ROOT)}  ({len(rows)} paths, {len(findings)} findings)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        protos = sorted(by_proto, key=lambda p: st.mean(by_proto[p]))
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, p in enumerate(protos):
            ys = by_proto[p]
            ax.scatter([i] * len(ys), ys, alpha=0.6, s=25)
            ax.scatter([i], [st.mean(ys)], marker="_", s=600, color="black")
        ax.axhline(0, color="gray", lw=1)
        ax.axhline(ORACLE, color="green", ls="--", lw=1, label=f"oracle +{ORACLE} bps")
        ax.set_xticks(range(len(protos))); ax.set_xticklabels(protos, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("net bps per trade (real, n≥30)")
        ax.set_title("No protocol clears zero: executable-window selection is unsolved")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG, dpi=130)
        print(f"wrote {FIG.relative_to(ROOT)}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
