import numpy as np
from wqrag import config as C
from wqrag.preprocessing import (chronological_split, clean_series, fit_standardiser, label_windows,
                                 make_windows, standardise)
from _synthetic import make_series


def test_clean_removes_sentinel_and_bounds():
    df = make_series(10)
    clean = clean_series(df)
    assert not (clean == C.USGS_SENTINEL).any().any()
    assert clean["pH"].max() <= 14


def test_split_is_chronological_and_proportional():
    df = clean_series(make_series(30))
    tr, va, te = chronological_split(df)
    assert tr.index.max() < va.index.min() < te.index.min()
    assert abs(len(tr) / len(df) - 0.70) < 0.01


def test_standardiser_uses_training_stats_only():
    df = clean_series(make_series(30))
    tr, va, te = chronological_split(df)
    mean, std = fit_standardiser(tr)
    z_tr = standardise(tr, mean, std)
    assert np.allclose(z_tr.mean(0), 0, atol=1e-5) and np.allclose(z_tr.std(0), 1, atol=1e-3)
    z_te = standardise(te, mean, std)
    assert not np.allclose(z_te.mean(0), 0, atol=1e-2)   # test uses train stats, not its own


def test_window_shape_and_stride():
    x = np.arange(96 * 10 * 5, dtype=float).reshape(960, 5)
    w = make_windows(x, 96, 24)
    assert w.shape == ((960 - 96) // 24 + 1, 96, 5)
    assert np.array_equal(w[1, 0], x[24])


def test_labels_three_sigma():
    w = np.zeros((4, 96, 5))
    w[1, 10, 2] = 3.5
    w[3, 0, 0] = -3.1
    assert label_windows(w).tolist() == [0, 1, 0, 1]
