"""Binance public historical archive downloader.

Downloads spot trades, aggTrades, klines, and bookTicker files from
``https://data.binance.vision`` and writes them to the Bronze vendor layer.

Reference:
    https://data.binance.vision
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

import requests

from stressbench.common.config import bronze_root
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_BASE = "https://data.binance.vision/data/spot/daily"
_DATA_TYPES = Literal["trades", "aggTrades", "klines", "bookTicker"]


def _archive_url(
    data_type: str,
    symbol: str,
    date: str,
    kline_interval: str = "1m",
) -> str:
    """Construct the download URL for a Binance archive file.

    Args:
        data_type: One of ``trades``, ``aggTrades``, ``klines``, ``bookTicker``.
        symbol: Binance symbol string (e.g. ``"USDCUSDT"``).
        date: ISO date string ``YYYY-MM-DD``.
        kline_interval: Kline interval (only used when ``data_type="klines"``).

    Returns:
        Full HTTPS URL to the ``.zip`` file.
    """
    sym = symbol.upper()
    if data_type == "klines":
        return f"{_BASE}/{data_type}/{sym}/{kline_interval}/{sym}-{kline_interval}-{date}.zip"
    return f"{_BASE}/{data_type}/{sym}/{sym}-{data_type}-{date}.zip"


def download_archive_file(
    data_type: str,
    symbol: str,
    date: str,
    root: Path | None = None,
    kline_interval: str = "1m",
    overwrite: bool = False,
) -> Path | None:
    """Download one Binance archive file and save it to the Bronze vendor layer.

    Args:
        data_type: One of ``trades``, ``aggTrades``, ``klines``, ``bookTicker``.
        symbol: Binance symbol string.
        date: ISO date string ``YYYY-MM-DD``.
        root: Bronze root override.
        kline_interval: Kline interval (only relevant for ``klines``).
        overwrite: If ``False`` and the file already exists, skip download.

    Returns:
        Path to the extracted CSV file, or ``None`` on failure.
    """
    root = root or bronze_root()
    out_dir = (
        root
        / "vendor=binance_archive"
        / f"data_type={data_type}"
        / f"symbol={symbol}"
        / f"date={date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{symbol}-{data_type}-{date}.csv"
    csv_path = out_dir / csv_name

    if csv_path.exists() and not overwrite:
        logger.info("Already exists, skipping: %s", csv_path)
        return csv_path

    url = _archive_url(data_type, symbol, date, kline_interval)
    logger.info("Downloading %s", url)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("HTTP error for %s: %s", url, exc)
        return None
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        csv_files = [n for n in names if n.endswith(".csv")]
        if not csv_files:
            logger.warning("No CSV found in archive: %s", url)
            return None
        zf.extract(csv_files[0], out_dir)
        extracted = out_dir / csv_files[0]
        if extracted != csv_path:
            extracted.rename(csv_path)

    logger.info("Saved %s", csv_path)
    return csv_path


def pull_event_window(
    symbol: str,
    start_date: str,
    end_date: str,
    data_types: list[str] | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Download all archive files for a symbol over a date range.

    Args:
        symbol: Binance symbol string.
        start_date: Start date ``YYYY-MM-DD`` (inclusive).
        end_date: End date ``YYYY-MM-DD`` (inclusive).
        data_types: List of data types to download; defaults to
            ``["trades", "aggTrades", "klines"]``.
        root: Bronze root override.

    Returns:
        List of paths to successfully downloaded CSV files.
    """
    from datetime import date, timedelta

    if data_types is None:
        data_types = ["trades", "aggTrades", "klines"]

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    paths: list[Path] = []
    current = start
    while current <= end:
        for dt in data_types:
            p = download_archive_file(
                data_type=dt,
                symbol=symbol,
                date=current.isoformat(),
                root=root,
            )
            if p:
                paths.append(p)
        current += timedelta(days=1)
    return paths
