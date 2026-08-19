import numpy as np
import pytest
from wqrag import config as C
from wqrag.detection import detection_metrics, run_isolation_forest, run_ocsvm, window_features
from wqrag.thresholding import fixed_threshold, select_threshold, threshold_sweep


def test_csi_and_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 0, 1, 1, 0])
    m = detection_metrics(y, p, scores=np.array([.9, .2, .1, .7, .8, .3]))
    assert m["TP"] == 2 and m["FP"] == 1 and m["FN"] == 1
    assert abs(m["CSI"] - 2 / 4) < 1e-9
    assert 0 <= m["AUC"] <= 1


def test_adaptive_threshold_prefers_precision_constraint():
    rng = np.random.RandomState(0)
    tr = rng.gamma(2, 0.01, 500)
    val = np.concatenate([rng.gamma(2, 0.01, 400), rng.gamma(2, 0.05, 100)])
    lab = np.r_[np.zeros(400), np.ones(100)].astype(int)
    ch = select_threshold(tr, val, lab)
    assert ch.val_precision >= C.THRESHOLD_MIN_PRECISION
    assert len(ch.candidates) == len(C.THRESHOLD_ALPHAS) + len(C.THRESHOLD_PERCENTILES)
    assert fixed_threshold(tr) > tr.mean()
    g, f1 = threshold_sweep(val, lab, 50)
    assert g.shape == f1.shape == (50,)


def test_baselines_run_with_paper_hyperparameters():
    rng = np.random.RandomState(1)
    tr = rng.normal(0, 1, (200, 96, 5)); te = rng.normal(0, 1, (50, 96, 5)); te[:5] += 4
    assert window_features(tr).shape == (200, 40)
    _, p1 = run_isolation_forest(tr, te); _, p2 = run_ocsvm(tr, te)
    assert p1.shape == p2.shape == (50,) and p1[:5].mean() > p1[5:].mean()
    assert C.ISOLATION_FOREST["n_estimators"] == 200 and C.ONE_CLASS_SVM["nu"] == 0.05


@pytest.mark.slow
def test_autoencoders_train_and_score():
    torch = pytest.importorskip("torch")
    from wqrag.models import build_model
    from wqrag.detection import score_windows, train_autoencoder
    rng = np.random.RandomState(2)
    tr = rng.normal(0, 1, (64, 96, 5)).astype(np.float32); va = tr[:16]
    cfg = dict(C.TRAINING, epochs=2, batch_size=32, patience=5)
    for name in ("PatchTST", "LSTM-AE"):
        m = build_model(name, 5)
        m, log = train_autoencoder(m, tr, va, name=name, cfg=cfg, device="cpu", verbose=False)
        e = score_windows(m, va, device="cpu")
        assert e.shape == (16,) and np.all(e >= 0) and len(log.train_loss) == 2
    pt = build_model("PatchTST", 5)
    assert pt.n_patches == 6 and pt.bottleneck.out_features == 64
