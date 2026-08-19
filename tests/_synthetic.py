"""Synthetic 15-min series with diurnal cycles + injected excursions for tests."""
import numpy as np
import pandas as pd
from wqrag import config as C


def make_series(n_days=60, seed=0, station="14211010"):
    rng = np.random.RandomState(seed)
    n = n_days * 96
    t = np.arange(n)
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    diurnal = np.sin(2 * np.pi * t / 96)
    df = pd.DataFrame({
        "Temperature_C": 12 + 3 * diurnal + rng.normal(0, 0.3, n),
        "Conductivity_uScm": 50 + 2 * diurnal + rng.normal(0, 1.0, n),
        "DO_mgL": 11 + 1.0 * diurnal + rng.normal(0, 0.2, n),
        "pH": 7.6 + 0.2 * diurnal + rng.normal(0, 0.05, n),
        "Turbidity_FNU": np.abs(3 + rng.normal(0, 1.0, n)),
    }, index=idx)
    # a few genuine excursions late in the record (test partition)
    for s in (int(n * 0.9), int(n * 0.93), int(n * 0.97)):
        df.iloc[s:s + 30, 4] += 150   # turbidity storm
        df.iloc[s:s + 30, 1] += 40    # SC perturbation
    # a sentinel and an out-of-bounds value
    df.iloc[10, 0] = C.USGS_SENTINEL
    df.iloc[20, 3] = 15.0
    return df
