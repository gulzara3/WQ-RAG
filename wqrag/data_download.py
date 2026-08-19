"""
Download 15-minute instantaneous values (IV) for the four USGS stations via the
NWIS REST API (Section 2.1; Data Availability statement).

Output: data/raw/station_<ID>.csv with a UTC datetime index and one column per
parameter (Table 2 codes).  A qualifier column `<param>_cd` is kept for
provenance but is dropped during preprocessing.

Usage
-----
    python scripts/01_download_data.py                 # all stations, 2021-2024
    python scripts/01_download_data.py --station 14211010 --start 2021-01-01 --end 2021-03-31

The IV service returns at most ~120 days per request reliably, so the range is
chunked by calendar month and concatenated.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from . import config as C
from .utils import get_logger

log = get_logger(__name__)

NWIS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def _month_chunks(start: str, end: str) -> Iterable[tuple[str, str]]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    cur = s
    while cur <= e:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        nxt = min(nxt, e)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + timedelta(days=1)


def _parse_rdb(text: str) -> pd.DataFrame:
    """Parse USGS RDB (tab-delimited with comment lines and a type row)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    header = lines[0].split("\t")
    body = "\n".join(lines[2:])  # skip the "5s 15s ..." type row
    df = pd.read_csv(io.StringIO(body), sep="\t", names=header, dtype=str)
    return df


def fetch_station(station_id: str, start: str = C.START_DATE, end: str = C.END_DATE,
                  pause: float = 0.5, retries: int = 3) -> pd.DataFrame:
    """Fetch all five parameters for one station over [start, end]."""
    codes = ",".join(C.PARAMETERS.keys())
    frames = []
    for s, e in _month_chunks(start, end):
        params = dict(format="rdb", sites=station_id, parameterCd=codes,
                      startDT=s, endDT=e, siteStatus="all")
        for attempt in range(retries):
            try:
                r = requests.get(NWIS_IV_URL, params=params, timeout=120)
                r.raise_for_status()
                df = _parse_rdb(r.text)
                if len(df):
                    frames.append(df)
                log.info("  %s  %s..%s  %6d rows", station_id, s, e, len(df))
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("  retry %d for %s %s..%s: %s", attempt + 1, station_id, s, e, exc)
                time.sleep(2 * (attempt + 1))
        time.sleep(pause)

    if not frames:
        raise RuntimeError(f"No data returned for station {station_id}")

    raw = pd.concat(frames, ignore_index=True)
    return _tidy(raw, station_id)


def _tidy(raw: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Map NWIS columns (e.g. '12345_00010') to friendly names; keep qualifiers."""
    ts = pd.to_datetime(raw["datetime"], errors="coerce", utc=False)
    out = pd.DataFrame(index=ts)
    for col in raw.columns:
        for code, name in C.PARAMETERS.items():
            if col.endswith(f"_{code}") and not col.endswith("_cd"):
                if name in out.columns:         # several TS ids for one code -> keep first
                    continue
                out[name] = pd.to_numeric(raw[col].values, errors="coerce")
                cd_col = f"{col}_cd"
                if cd_col in raw.columns:
                    out[f"{name}_cd"] = raw[cd_col].values
    out.index.name = "datetime"
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="first")]
    log.info("Station %s: %d rows, params=%s", station_id,
             len(out), [c for c in out.columns if not c.endswith("_cd")])
    return out


def download_all(stations: Iterable[str] = C.STATION_ORDER, start: str = C.START_DATE,
                 end: str = C.END_DATE, out_dir: Path = C.RAW_DATA_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sid in stations:
        log.info("Downloading USGS %s (%s)", sid, C.STATIONS[sid]["name"])
        df = fetch_station(sid, start, end)
        p = out_dir / f"station_{sid}.csv"
        df.to_csv(p)
        paths[sid] = p
        log.info("Saved %s", p)
    return paths
