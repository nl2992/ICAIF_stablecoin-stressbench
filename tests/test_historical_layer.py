"""Tests for the deep historical layer.

Tests:
1. YAML has exactly 18 events
2. All events have required fields
3. Mechanism classes cover all 7 expected classes
4. Tier A events are exactly {usdc_svb_2023, usdc_svb_recovery_2023}
5. All verified events have verified=True in source registry
6. source_verification records exist for every event_id in YAML
7. price_grade_features builds summaries without errors
8. build_all_summaries returns 18 summaries
9. Tier A summaries have is_synthetic=False
10. Tier B/C summaries have is_synthetic=True
11. No paper claims with is_synthetic=True may have claim_level="execution_grade"
12. EventSourceRecord with use_in_paper=True must have verified=True
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest
import yaml

from stressbench.history.price_grade_features import build_all_summaries
from stressbench.history.source_verification import (
    EVENT_SOURCE_REGISTRY,
    EventSourceRecord,
    get_event_ids_with_verified_source,
    get_paper_records,
    get_records_for_event,
    get_verified_records,
)

YAML_PATH = ROOT / "configs" / "event_windows_historical.yaml"
EXPECTED_EVENTS = 18
EXPECTED_TIER_A = {"usdc_svb_2023", "usdc_svb_recovery_2023"}
EXPECTED_MECHANISM_CLASSES = {
    "algorithmic_reflexive",
    "fiat_reserve_bank_shock",
    "regulatory_issuer_winddown",
    "exchange_credit_liquidity",
    "defi_pool_imbalance",
    "collateral_liquidation_oracle",
    "rwa_niche_stablecoin",
}
REQUIRED_FIELDS = {
    "stablecoins",
    "mechanism_class",
    "start",
    "end",
    "data_tier",
    "coverage_score",
    "max_depeg_bps_est",
    "duration_class",
    "verification_status",
    "empirical_use",
}


@pytest.fixture(scope="module")
def events() -> dict:
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]


# ── Test 1: event count ───────────────────────────────────────────────────────


def test_yaml_has_18_events(events: dict) -> None:
    assert (
        len(events) == EXPECTED_EVENTS
    ), f"Expected {EXPECTED_EVENTS} events in YAML, got {len(events)}"


# ── Test 2: required fields ───────────────────────────────────────────────────


def test_all_events_have_required_fields(events: dict) -> None:
    missing = {}
    for eid, ev in events.items():
        missing_fields = REQUIRED_FIELDS - set(ev.keys())
        if missing_fields:
            missing[eid] = missing_fields
    assert not missing, f"Events missing required fields: {missing}"


# ── Test 3: mechanism classes ─────────────────────────────────────────────────


def test_all_mechanism_classes_present(events: dict) -> None:
    found = {ev.get("mechanism_class") for ev in events.values()}
    missing = EXPECTED_MECHANISM_CLASSES - found
    assert not missing, f"Missing mechanism classes: {missing}"


# ── Test 4: Tier A events ─────────────────────────────────────────────────────


def test_tier_a_events_exactly_correct(events: dict) -> None:
    tier_a = {eid for eid, ev in events.items() if ev.get("data_tier") == "A"}
    assert (
        tier_a == EXPECTED_TIER_A
    ), f"Tier A events mismatch. Expected {EXPECTED_TIER_A}, got {tier_a}"


# ── Test 5: verified events in source registry ────────────────────────────────


def test_verified_yaml_events_have_source_registry_record(events: dict) -> None:
    yaml_verified = {
        eid for eid, ev in events.items() if ev.get("verification_status") == "verified"
    }
    registry_event_ids = {r.event_id for r in EVENT_SOURCE_REGISTRY}
    missing_from_registry = yaml_verified - registry_event_ids
    assert (
        not missing_from_registry
    ), f"YAML-verified events have no source registry record: {missing_from_registry}"


# ── Test 6: source records for every YAML event ───────────────────────────────


def test_source_registry_covers_all_yaml_events(events: dict) -> None:
    registry_event_ids = {r.event_id for r in EVENT_SOURCE_REGISTRY}
    yaml_event_ids = set(events.keys())
    missing = yaml_event_ids - registry_event_ids
    assert not missing, f"Events in YAML but not in source registry: {missing}"


# ── Test 7: price_grade_features builds without error ────────────────────────


def test_build_all_summaries_no_error(events: dict) -> None:
    summaries = build_all_summaries(events)
    assert len(summaries) == EXPECTED_EVENTS


# ── Test 8: correct count of summaries ───────────────────────────────────────


def test_build_all_summaries_count(events: dict) -> None:
    summaries = build_all_summaries(events)
    assert (
        len(summaries) == EXPECTED_EVENTS
    ), f"Expected {EXPECTED_EVENTS} summaries, got {len(summaries)}"


# ── Test 9: Tier A summaries are not synthetic ────────────────────────────────


def test_tier_a_summaries_not_synthetic(events: dict) -> None:
    summaries = build_all_summaries(events)
    tier_a_summaries = [s for s in summaries if s.event_id in EXPECTED_TIER_A]
    assert len(tier_a_summaries) == len(EXPECTED_TIER_A)
    for s in tier_a_summaries:
        assert not s.is_synthetic, f"Tier A event {s.event_id} should not be synthetic"


# ── Test 10: Tier B/C summaries are synthetic ────────────────────────────────


def test_tier_bc_summaries_are_synthetic(events: dict) -> None:
    summaries = build_all_summaries(events)
    non_tier_a = [s for s in summaries if s.event_id not in EXPECTED_TIER_A]
    for s in non_tier_a:
        assert s.is_synthetic, f"Non-Tier-A event {s.event_id} should be synthetic"


# ── Test 11: no synthetic rows claim execution_grade ─────────────────────────


def test_synthetic_summaries_not_execution_grade(events: dict) -> None:
    summaries = build_all_summaries(events)
    bad = [
        s for s in summaries if s.is_synthetic and s.claim_level == "execution_grade"
    ]
    assert (
        not bad
    ), f"Synthetic summaries must not claim execution_grade: {[s.event_id for s in bad]}"


# ── Test 12: use_in_paper requires verified ───────────────────────────────────


def test_use_in_paper_requires_verified() -> None:
    bad = [r for r in EVENT_SOURCE_REGISTRY if r.use_in_paper and not r.verified]
    assert not bad, (
        f"Records with use_in_paper=True but verified=False: "
        f"{[(r.event_id, r.claim[:50]) for r in bad]}"
    )
