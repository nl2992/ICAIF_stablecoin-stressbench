"""Configuration loader: reads YAML config files and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIGS_DIR / name
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_venues() -> dict[str, Any]:
    return _load_yaml("venues.yaml")["venues"]


def load_instruments() -> list[dict[str, Any]]:
    return _load_yaml("instruments.yaml")["instruments"]


def load_event_windows() -> dict[str, Any]:
    return _load_yaml("event_windows.yaml")["events"]


def load_fee_schedules() -> dict[str, Any]:
    return _load_yaml("fee_schedules.yaml")


def load_token_addresses() -> dict[str, Any]:
    return _load_yaml("token_addresses.yaml")["tokens"]


def load_benchmark_splits() -> dict[str, Any]:
    return _load_yaml("benchmark_splits.yaml")


def get_env(key: str, default: str | None = None) -> str | None:
    """Return an environment variable value, with optional default."""
    return os.environ.get(key, default)


def bronze_root() -> Path:
    return Path(get_env("BRONZE_DATA_DIR", "./data/bronze"))


def silver_root() -> Path:
    return Path(get_env("SILVER_DATA_DIR", "./data/silver"))


def gold_root() -> Path:
    return Path(get_env("GOLD_DATA_DIR", "./data/gold"))
