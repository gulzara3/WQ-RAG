"""
Stage II — train and evaluate the four detectors on one station.

Models (Section 2.2.2): PatchTST-AE, LSTM-AE (AdamW, lr 1e-4, wd 1e-5, batch 64,
<=100 epochs, early stopping patience 10, ReduceLROnPlateau), Isolation Forest
(200 trees, contamination 0.05) and One-Class SVM (RBF, nu 0.05, gamma 'scale').

Metrics (Section 2.4.2): Precision, Recall, F1, CSI = TP/(TP+FP+FN), AUC-ROC for
continuous-score models, and full confusion matrices (Table 3, Fig. 3, Fig. 9).

Outputs per station (results/):
    detection_<sid>.json      metrics, thresholds, confusion counts, loss curves
    scores_<sid>.npz          per-window scores + labels (needed for Figs 5, 6)
    models/<model>_<sid>.pt   trained weights
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import OneClassSVM

from . import config as C
from .preprocessing import StationData
from .thresholding import ThresholdChoice, fixed_threshold, select_threshold
from .utils import get_device, get_logger, save_json, set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      scores: Optional[np.ndarray] = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0            # Eq. (1)
    out = dict(Precision=p, Recall=r, F1=f1, CSI=csi, TP=int(tp), FP=int(fp),
               FN=int(fn), TN=int(tn), Accuracy=(tp + tn) / max(len(y_true), 1))
    if scores is not None:
        try:
            out["AUC"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            out["AUC"] = float("nan")
    else:
        out["AUC"] = None
    return out


# ---------------------------------------------------------------------------
# Deep-learning training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainLog:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def train_autoencoder(model, train: np.ndarray, val: np.ndarray, name: str = "model",
                      cfg: dict = C.TRAINING, device: Optional[str] = None,
                      verbose: bool = True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = device or get_device()
    model = model.to(device)
    x_tr = torch.tensor(train, dtype=torch.float32)
    x_va = torch.tensor(val, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(x_tr), batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=cfg["scheduler_patience"], factor=cfg["scheduler_factor"])
    crit = torch.nn.MSELoss()

    tl = TrainLog()
    best, best_state, wait, t0 = float("inf"), None, 0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        run = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            run += loss.item() * len(xb)
        tr_loss = run / len(x_tr)

        model.eval()
        with torch.no_grad():
            va_loss = crit(model(x_va), x_va).item()
        sched.step(va_loss)
        tl.train_loss.append(tr_loss)
        tl.val_loss.append(va_loss)

        if va_loss < best - 1e-7:
            best, wait, tl.best_epoch = va_loss, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            log.info("    %s epoch %3d | train %.5f | val %.5f", name, epoch + 1, tr_loss, va_loss)
        if wait >= cfg["patience"]:
            if verbose:
                log.info("    %s early stopping at epoch %d", name, epoch + 1)
            break

    model.load_state_dict(best_state)
    tl.seconds = time.time() - t0
    return model.to(device), tl


def score_windows(model, windows: np.ndarray, batch: int = 256, device: Optional[str] = None) -> np.ndarray:
    import torch
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.tensor(windows[i:i + batch], dtype=torch.float32).to(device)
            out.append(model.reconstruction_error(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# Conventional baselines (window statistics as features)
# ---------------------------------------------------------------------------
def window_features(windows: np.ndarray) -> np.ndarray:
    """Per-parameter summary statistics of each window (8 x 5 = 40 features)."""
    mn = windows.mean(1); sd = windows.std(1)
    mi = windows.min(1); ma = windows.max(1)
    q25 = np.percentile(windows, 25, axis=1); q75 = np.percentile(windows, 75, axis=1)
    rng = ma - mi
    mac = np.abs(np.diff(windows, axis=1)).mean(1)
    return np.concatenate([mn, sd, mi, ma, q25, q75, rng, mac], axis=1)


def run_isolation_forest(train: np.ndarray, test: np.ndarray):
    clf = IsolationForest(**C.ISOLATION_FOREST).fit(window_features(train))
    pred = (clf.predict(window_features(test)) == -1).astype(int)
    return clf, pred


def run_ocsvm(train: np.ndarray, test: np.ndarray, seed: int = C.SEED):
    x = window_features(train)
    if len(x) > C.OCSVM_MAX_TRAIN:
        idx = np.random.RandomState(seed).choice(len(x), C.OCSVM_MAX_TRAIN, replace=False)
        x = x[idx]
    clf = OneClassSVM(**C.ONE_CLASS_SVM).fit(x)
    pred = (clf.predict(window_features(test)) == -1).astype(int)
    return clf, pred


# ---------------------------------------------------------------------------
# Per-station orchestration
# ---------------------------------------------------------------------------
def run_station(sd: StationData, models: tuple = ("PatchTST", "LSTM-AE", "IF", "OC-SVM"),
                save_dir: Path = C.RESULTS_DIR, train_cfg: dict = C.TRAINING,
                verbose: bool = True) -> dict:
    """Train/evaluate all detectors on one station and persist results."""
    from .models import build_model
    set_seed(C.SEED)
    sid = sd.station_id
    results: dict = {"station_id": sid, "name": sd.name, "models": {}, "n_test": int(len(sd.test))}
    score_store = {"labels": sd.test_labels}

    for name in models:
        log.info("==> %s | %s", sd.name, name)
        if name in ("PatchTST", "LSTM-AE"):
            model = build_model(name, sd.n_features)
            model, tl = train_autoencoder(model, sd.train, sd.val, name=name, cfg=train_cfg, verbose=verbose)
            e_tr, e_va, e_te = (score_windows(model, w) for w in (sd.train, sd.val, sd.test))
            choice: ThresholdChoice = select_threshold(e_tr, e_va, sd.val_labels)
            theta_fixed = fixed_threshold(e_tr)
            pred = (e_te > choice.threshold).astype(int)
            m = detection_metrics(sd.test_labels, pred, scores=e_te)
            m_fixed = detection_metrics(sd.test_labels, (e_te > theta_fixed).astype(int))
            m.update(threshold=choice.threshold, threshold_rule=choice.rule,
                     fixed_threshold=theta_fixed, F1_fixed_threshold=m_fixed["F1"],
                     val_f1=choice.val_f1, best_epoch=tl.best_epoch, train_seconds=tl.seconds,
                     train_loss=tl.train_loss, val_loss=tl.val_loss,
                     threshold_candidates=choice.candidates)
            score_store[f"{name}_test"] = e_te
            score_store[f"{name}_train"] = e_tr
            score_store[f"{name}_val"] = e_va
            try:
                import torch
                (save_dir / "models").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "models" / f"{name}_{sid}.pt")
            except Exception as exc:  # noqa: BLE001
                log.warning("could not save model weights: %s", exc)
        elif name == "IF":
            _, pred = run_isolation_forest(sd.train, sd.test)
            m = detection_metrics(sd.test_labels, pred)
        elif name == "OC-SVM":
            _, pred = run_ocsvm(sd.train, sd.test)
            m = detection_metrics(sd.test_labels, pred)
        else:
            raise KeyError(name)
        score_store[f"{name}_pred"] = pred
        results["models"][name] = m
        log.info("    P=%.3f R=%.3f F1=%.3f CSI=%.3f AUC=%s",
                 m["Precision"], m["Recall"], m["F1"], m["CSI"],
                 f"{m['AUC']:.3f}" if m.get("AUC") is not None else "—")

    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, save_dir / f"detection_{sid}.json")
    np.savez_compressed(save_dir / f"scores_{sid}.npz", **score_store)
    return results


def results_to_table3(all_results: dict[str, dict]) -> pd.DataFrame:
    """Long-format Table 3 (station x model)."""
    rows = []
    for sid in C.STATION_ORDER:
        if sid not in all_results:
            continue
        for model in C.MODEL_ORDER:
            m = all_results[sid]["models"].get(model)
            if not m:
                continue
            rows.append(dict(Station=C.STATIONS[sid]["name"], station_id=sid, Model=model,
                             Precision=m["Precision"], Recall=m["Recall"], F1=m["F1"],
                             CSI=m["CSI"], TP=m["TP"], FP=m["FP"], FN=m["FN"], TN=m["TN"],
                             AUC=m.get("AUC")))
    return pd.DataFrame(rows)
