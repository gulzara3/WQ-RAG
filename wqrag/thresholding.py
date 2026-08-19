"""
Adaptive per-station thresholding (Section 2.2.2, last paragraph; Fig. 2).

Candidate thresholds:
    statistical  theta = mu(e_train) + alpha * sigma(e_train),  alpha in {1.0 .. 4.0}
    percentile   theta = P_q(e_train),                          q in {90, 92, 95, 97, 99}
The candidate maximising validation F1 subject to precision >= 0.65 is chosen.
The selection is applied identically at every station.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class ThresholdChoice:
    threshold: float
    rule: str            # e.g. "alpha=2.5" or "P95"
    val_f1: float
    val_precision: float
    val_recall: float
    candidates: list     # [(rule, theta, P, R, F1), ...] for Fig. 5d / audit


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fixed_threshold(train_scores: np.ndarray, alpha: float = C.FIXED_ALPHA) -> float:
    """Single fixed rule theta = mu + alpha*sigma (reference baseline)."""
    return float(train_scores.mean() + alpha * train_scores.std())


def select_threshold(train_scores: np.ndarray, val_scores: np.ndarray, val_labels: np.ndarray,
                     alphas=C.THRESHOLD_ALPHAS, percentiles=C.THRESHOLD_PERCENTILES,
                     min_precision: float = C.THRESHOLD_MIN_PRECISION) -> ThresholdChoice:
    mu, sd = float(train_scores.mean()), float(train_scores.std())
    candidates = [(f"alpha={a:.1f}", mu + a * sd) for a in alphas]
    candidates += [(f"P{q}", float(np.percentile(train_scores, q))) for q in percentiles]

    evaluated = []
    best = None
    for rule, theta in candidates:
        p, r, f = _prf(val_labels, (val_scores > theta).astype(int))
        evaluated.append((rule, theta, p, r, f))
        if p >= min_precision and (best is None or f > best[4]):
            best = (rule, theta, p, r, f)

    if best is None:                      # precision constraint never met: fall back to best F1
        best = max(evaluated, key=lambda t: t[4])

    rule, theta, p, r, f = best
    return ThresholdChoice(threshold=float(theta), rule=rule, val_f1=f,
                           val_precision=p, val_recall=r, candidates=evaluated)


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """F1 as a function of theta on a log grid (Fig. 5d)."""
    pos = scores[scores > 0]
    lo, hi = np.log10(pos.min() + 1e-12), np.log10(pos.max())
    grid = np.logspace(lo - 0.5, hi + 0.2, n)
    f1 = np.array([_prf(labels, (scores > t).astype(int))[2] for t in grid])
    return grid, f1
