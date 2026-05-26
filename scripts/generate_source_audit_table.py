"""Generate source audit table from EVENT_SOURCE_REGISTRY.

Writes: results/paper_addon/table_16_event_source_audit.csv

Each row is one EventSourceRecord. Columns:
  event_id, claim_short, source_type, source_name, verified, use_in_paper, notes_short
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv

from stressbench.history.source_verification import EVENT_SOURCE_REGISTRY

OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "table_16_event_source_audit.csv"

FIELDNAMES = [
    "event_id",
    "claim_short",
    "source_type",
    "source_name",
    "url",
    "verified",
    "use_in_paper",
    "notes_short",
]


def truncate(s: str, maxlen: int = 80) -> str:
    s = s.strip().replace("\n", " ")
    return s[:maxlen] + "…" if len(s) > maxlen else s


def main() -> None:
    rows = []
    for rec in EVENT_SOURCE_REGISTRY:
        rows.append(
            {
                "event_id": rec.event_id,
                "claim_short": truncate(rec.claim, 70),
                "source_type": rec.source_type,
                "source_name": truncate(rec.source_name, 80),
                "url": rec.url,
                "verified": str(rec.verified),
                "use_in_paper": str(rec.use_in_paper),
                "notes_short": truncate(rec.notes, 100),
            }
        )

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    verified_n = sum(1 for r in EVENT_SOURCE_REGISTRY if r.verified)
    paper_n = sum(1 for r in EVENT_SOURCE_REGISTRY if r.use_in_paper)
    events_n = len({r.event_id for r in EVENT_SOURCE_REGISTRY})
    print(f"Wrote {len(rows)} source records for {events_n} events → {OUT_FILE}")
    print(f"  verified={verified_n}, use_in_paper={paper_n}")


if __name__ == "__main__":
    main()
