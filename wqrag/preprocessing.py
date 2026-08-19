"""
Stage I — Data preprocessing (Section 2.2.1; Fig. 2 Stage I).

Pipeline (identical for all stations):
  1. Replace USGS sentinel values (-999,999) with NaN.
  2. Physical-bounds filtering (T -5..45 °C, pH 2..14, DO 0..25 mg/L,
     turbidity 0..5,000 FNU).
  3. Forward-fill gaps of <= 1 h (4 steps); drop longer gaps.
  4. Chronological 70/15/15 split (train / val / test) -> no leakage.
  5. Z-score standardisation with TRAINING-SET statistics only.
  6. Sliding windows of 96 steps (24 h) with stride 24 (6 h) -> x in R^{96x5}.
  7. Ground-truth labels: a window is anomalous if ANY parameter at ANY step
     deviates by more than ±3 sigma from the training-period mean.
     (Thresholds are computed on the training partition alone.)

The returned `StationData` object is what every downstream stage consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from .utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class StationData:
    station_id: str
    name: str
    columns: list[str]
    mean: np.ndarray            # training-period mean (per parameter)
    std: np.ndarray             # training-period std  (per parameter)
    train: np.ndarray           # (n_train, 96, 5) standardised windows
    val: np.ndarray
    test: np.ndarray
    train_labels: np.ndarray    # ±3σ statistical labels (for reference only)
    val_labels: np.ndarray      # used for adaptive threshold selection
    test_labels: np.ndarray     # used for final evaluation
    test_index: pd.DatetimeIndex  # timestamps of the test partition (raw rows)
    test_series: np.ndarray     # standardised test partition (n_steps, 5)
    n_raw_rows: int = 0
    n_clean_rows: int = 0
    extras: dict = field(default_factory=dict)

    # --- convenience -------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.columns)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Map standardised values back to physical units."""
        return z * self.std + self.mean

    def summary(self) -> dict:
        return dict(
            station_id=self.station_id, name=self.name,
            n_raw_rows=self.n_raw_rows, n_clean_rows=self.n_clean_rows,
            n_train=len(self.train), n_val=len(self.val), n_test=len(self.test),
            test_anomaly_rate=float(self.test_labels.mean()),
            columns=self.columns,
        )

    # --- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, columns=np.array(self.columns), mean=self.mean, std=self.std,
            train=self.train, val=self.val, test=self.test,
            train_labels=self.train_labels, val_labels=self.val_labels,
            test_labels=self.test_labels,
            test_index=self.test_index.values.astype("datetime64[ns]"),
            test_series=self.test_series,
            meta=np.array([self.station_id, self.name, self.n_raw_rows, self.n_clean_rows], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "StationData":
        z = np.load(path, allow_pickle=True)
        meta = z["meta"]
        return cls(
            station_id=str(meta[0]), name=str(meta[1]), columns=list(z["columns"]),
            mean=z["mean"], std=z["std"], train=z["train"], val=z["val"], test=z["test"],
            train_labels=z["train_labels"], val_labels=z["val_labels"], test_labels=z["test_labels"],
            test_index=pd.DatetimeIndex(z["test_index"]), test_series=z["test_series"],
            n_raw_rows=int(meta[2]), n_clean_rows=int(meta[3]),
        )


# ---------------------------------------------------------------------------
# Step functions (each is individually unit-tested)
# ---------------------------------------------------------------------------
def load_raw_station(station_id: str, raw_dir: Path = C.RAW_DATA_DIR) -> pd.DataFrame:
    """Load data/raw/station_<id>.csv and keep the five parameter columns."""
    path = raw_dir / f"station_{station_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/01_download_data.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    # Accept either friendly names or raw NWIS "<ts>_<code>" names.
    rename = {}
    for col in df.columns:
        if col.endswith("_cd"):
            continue
        for code, name in C.PARAMETERS.items():
            if col == name or col.endswith(f"_{code}"):
                rename[col] = name
    df = df.rename(columns=rename)
    cols = [c for c in C.PARAM_ORDER if c in df.columns]
    missing = set(C.PARAM_ORDER) - set(cols)
    if missing:
        raise ValueError(f"Station {station_id} is missing parameters {sorted(missing)}")
    return df[C.PARAM_ORDER].apply(pd.to_numeric, errors="coerce")


def clean_series(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel removal, physical-bounds filtering and gap handling."""
    df = df.copy()
    df = df.replace(C.USGS_SENTINEL, np.nan)
    for col, (lo, hi) in C.PHYSICAL_BOUNDS.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
    # Regularise to the 15-min grid so that 'steps' == time
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.asfreq(f"{C.SAMPLING_MINUTES}min")
    df = df.ffill(limit=C.MAX_GAP_STEPS)        # gaps <= 1 h
    df = df.dropna(how="any")                   # longer gaps are excluded
    return df


def chronological_split(df: pd.DataFrame,
                        ratios=(C.TRAIN_RATIO, C.VAL_RATIO, C.TEST_RATIO)):
    n = len(df)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va], df.iloc[n_tr + n_va:]


def fit_standardiser(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = train_df.mean().values.astype(np.float64)
    std = train_df.std(ddof=0).values.astype(np.float64)
    std[std == 0] = 1.0
    return mean, std


def standardise(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((df.values - mean) / std).astype(np.float32)


def make_windows(series: np.ndarray, window: int = C.WINDOW_SIZE,
                 stride: int = C.WINDOW_STRIDE) -> np.ndarray:
    """(n_steps, n_feat) -> (n_windows, window, n_feat) via stride_tricks."""
    n_steps = series.shape[0]
    if n_steps < window:
        return np.empty((0, window, series.shape[1]), dtype=series.dtype)
    n_win = (n_steps - window) // stride + 1
    idx = np.arange(window)[None, :] + stride * np.arange(n_win)[:, None]
    return series[idx]


def window_start_indices(n_steps: int, window: int = C.WINDOW_SIZE,
                         stride: int = C.WINDOW_STRIDE) -> np.ndarray:
    n_win = max((n_steps - window) // stride + 1, 0)
    return stride * np.arange(n_win)


def label_windows(windows: np.ndarray, sigma: float = C.LABEL_SIGMA) -> np.ndarray:
    """Statistical exceedance labels on *standardised* windows: any |z| > sigma."""
    if len(windows) == 0:
        return np.zeros(0, dtype=int)
    return (np.abs(windows).max(axis=(1, 2)) > sigma).astype(int)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare_station(station_id: str, raw_dir: Path = C.RAW_DATA_DIR) -> StationData:
    info = C.STATIONS[station_id]
    log.info("Preprocessing %s (%s)", station_id, info["name"])
    raw = load_raw_station(station_id, raw_dir)
    n_raw = len(raw)
    clean = clean_series(raw)
    log.info("  raw rows=%d  clean rows=%d", n_raw, len(clean))

    tr_df, va_df, te_df = chronological_split(clean)
    mean, std = fit_standardiser(tr_df)
    tr, va, te = (standardise(d, mean, std) for d in (tr_df, va_df, te_df))

    w_tr, w_va, w_te = (make_windows(s) for s in (tr, va, te))
    y_tr, y_va, y_te = (label_windows(w) for w in (w_tr, w_va, w_te))
    log.info("  windows train=%d val=%d test=%d | test anomaly rate=%.3f",
             len(w_tr), len(w_va), len(w_te), y_te.mean() if len(y_te) else 0)

    return StationData(
        station_id=station_id, name=info["name"], columns=list(clean.columns),
        mean=mean, std=std, train=w_tr, val=w_va, test=w_te,
        train_labels=y_tr, val_labels=y_va, test_labels=y_te,
        test_index=te_df.index, test_series=te,
        n_raw_rows=n_raw, n_clean_rows=len(clean),
    )


def prepare_all(stations=C.STATION_ORDER, raw_dir: Path = C.RAW_DATA_DIR,
                cache_dir: Optional[Path] = C.PROCESSED_DIR,
                use_cache: bool = True) -> dict[str, StationData]:
    out = {}
    for sid in stations:
        cache = (cache_dir / f"station_{sid}.npz") if cache_dir else None
        if use_cache and cache and cache.exists():
            log.info("Loading cached windows for %s", sid)
            out[sid] = StationData.load(cache)
            continue
        sd = prepare_station(sid, raw_dir)
        if cache:
            sd.save(cache)
        out[sid] = sd
    return out


def station_table(all_data: dict[str, StationData]) -> pd.DataFrame:
    """Table 1 — hydrogeochemical characteristics."""
    rows = []
    for sid, sd in all_data.items():
        info = C.STATIONS[sid]
        rows.append(dict(
            USGS_ID=sid, Station=info["name"], Region=info["region"],
            Raw_Rows=sd.n_raw_rows, Clean_Rows=sd.n_clean_rows,
            SC_range_uScm=info["sc_range"],
        ))
    return pd.DataFrame(rows)
