"""
Paper figures.  One function per figure, named after the manuscript numbering.

    fig01_station_map          Fig. 1  (cartopy optional; falls back to plain axes)
    fig03_detection_performance Fig. 3  (P / R / F1 / CSI, 4 stations x 4 models)
    fig04_training_curves      Fig. 4  (PatchTST & LSTM-AE losses, Clackamas)
    fig05_reconstruction_error Fig. 5  (violin, ROC, log-KDE Clackamas, threshold sweep)
    fig06_timeseries           Fig. 6  (Clackamas five parameters + error, anomalies shaded)
    fig07_explanation_quality  Fig. 7  (heatmap, radar, grouped bars, cross-station mean)
    fig08_llm_comparison       Fig. 8  (mean bars + paired differences box plots)
    fig09_confusion_matrices   Fig. 9  (4 models x 4 stations, counts + %)
    fig10_best_model           Fig. 10 (best F1 per station)

Fig. 2 (system architecture) is a hand-drawn diagram: docs/figures/fig02_system_architecture.png.

Functions take tidy DataFrames (Table 3-style long tables etc.) so they can be
driven either by a fresh run (results/) or by the published numbers
(paper_results/).  See scripts/09_make_figures_tables.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
    "axes.labelweight": "bold", "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

STATION_COLORS = {sid: C.STATIONS[sid]["color"] for sid in C.STATION_ORDER}
STATION_LABELS = {sid: C.STATIONS[sid]["label"] for sid in C.STATION_ORDER}
_MODEL_DISPLAY = {"PatchTST": "PatchTST", "LSTM-AE": "LSTM-AE", "IF": "IF", "OC-SVM": "OC-SVM",
                  "Isolation Forest": "IF"}


def _save(fig, name: str, out_dir: Path, formats=("png", "pdf")) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ===========================================================================
# Fig. 1 — station map
# ===========================================================================
def fig01_station_map(out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        have_cartopy = True
    except ImportError:
        have_cartopy = False

    fig = plt.figure(figsize=(12, 7.2))
    if have_cartopy:
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96, central_latitude=39))
        ax.set_extent([-125, -66, 23, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#dfeaf5")
        ax.add_feature(cfeature.LAKES, facecolor="#dfeaf5", edgecolor="#9fb9d1", linewidth=0.4)
        ax.add_feature(cfeature.RIVERS, edgecolor="#9fb9d1", linewidth=0.4)
        ax.add_feature(cfeature.STATES, edgecolor="#bbbbbb", linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, edgecolor="#666666", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#666666", linewidth=0.6)
        tr = ccrs.PlateCarree()
        ax.grid(False)
    else:
        ax = plt.axes()
        ax.set_xlim(-126, -66); ax.set_ylim(23, 50)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        tr = None

    offsets = {"01646500": (1.5, -2.5), "03351000": (-1.5, -3.0), "11447650": (1.8, -3.5), "14211010": (2.0, 1.5)}
    for sid in C.STATION_ORDER:
        s = C.STATIONS[sid]
        kw = {"transform": tr} if tr is not None else {}
        ax.scatter(s["lon"], s["lat"], s=140, color=s["color"], edgecolor="white", linewidth=1.2, zorder=5, **kw)
        dx, dy = offsets[sid]
        ax.text(s["lon"] + dx, s["lat"] + dy,
                f"{s['name']}\nUSGS {sid}\n{s['regime']}\n{s['sc_range']} µS/cm",
                fontsize=8.5, ha="left" if dx > 0 else "right", va="top", zorder=6, **kw)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=C.STATIONS[s]["color"],
                          label=f"{C.STATIONS[s]['name'].split(',')[0]}: {C.STATIONS[s]['sc_range']} µS/cm")
               for s in C.STATION_ORDER]
    ax.legend(handles=handles, title="Monitoring station (conductivity range)", loc="lower left", fontsize=9)
    ax.set_title("USGS continuous water-quality monitoring stations used in this study", fontsize=13)
    return _save(fig, "fig01_station_map", out_dir)


# ===========================================================================
# Fig. 3 — cross-station detection performance
# ===========================================================================
def fig03_detection_performance(table3: pd.DataFrame, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    metrics = [("Precision", "(a) Precision"), ("Recall", "(b) Recall"),
               ("F1", "(c) F1-Score"), ("CSI", "(d) Critical Success Index (CSI)")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharey=True)
    x = np.arange(len(C.MODEL_ORDER)); w = 0.2
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for i, sid in enumerate(C.STATION_ORDER):
            vals = []
            for m in C.MODEL_ORDER:
                r = table3[(table3["station_id"].astype(str) == sid) & (table3["Model"] == m)]
                vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
            bars = ax.bar(x + (i - 1.5) * w, vals, w, color=STATION_COLORS[sid], label=STATION_LABELS[sid],
                          edgecolor="white")
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=7, color=STATION_COLORS[sid], fontweight="bold")
        ax.axhline(0.8, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title); ax.set_ylim(0, 1.12); ax.set_xticks(x); ax.set_xticklabels(C.MODEL_ORDER, fontweight="bold")
        ax.set_ylabel("Score")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig03_detection_performance", out_dir)


# ===========================================================================
# Fig. 4 — training curves (Clackamas)
# ===========================================================================
def fig04_training_curves(detection_json: dict, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, tag in zip(axes, ["PatchTST", "LSTM-AE"], ["(a)", "(b)"]):
        m = detection_json["models"].get(name, {})
        tr, va = m.get("train_loss", []), m.get("val_loss", [])
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color="#1b3a5c", lw=1.6, label=f"{name} Train")
        ax.plot(ep, va, color="#e04b3c", lw=1.6, ls="--", label=f"{name} Val")
        ax.set_title(f"{tag}  {name}"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "fig04_training_curves", out_dir)


# ===========================================================================
# Fig. 5 — reconstruction error analysis (PatchTST)
# ===========================================================================
def _kde(x: np.ndarray, grid: np.ndarray, bw: Optional[float] = None) -> np.ndarray:
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.ptp(x) < 1e-12:          # degenerate -> narrow gaussian bump
        if len(x) == 0:
            return np.zeros_like(grid)
        w = max(np.ptp(grid) * 0.01, 1e-9)
        return np.exp(-0.5 * ((grid - x.mean()) / w) ** 2)
    try:
        return gaussian_kde(x, bw_method=bw)(grid)
    except np.linalg.LinAlgError:
        x = x + np.random.RandomState(0).normal(0, 1e-6 * max(abs(x).max(), 1), len(x))
        return gaussian_kde(x, bw_method=bw)(grid)


def fig05_reconstruction_error(scores: Dict[str, dict], out_dir: Path = C.FIGURES_DIR,
                               detector: str = "PatchTST", primary: str = C.PRIMARY_STATION) -> list[Path]:
    """scores[sid] = {"test": e, "labels": y, "threshold": theta}"""
    from sklearn.metrics import roc_curve, auc
    from .thresholding import threshold_sweep

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    sids = [s for s in C.STATION_ORDER if s in scores]

    # (a) split violins ------------------------------------------------------
    ax = axes[0, 0]
    ymax = 0.2
    for i, sid in enumerate(sids):
        e, y = scores[sid]["test"], scores[sid]["labels"]
        grid = np.linspace(0, ymax, 300)
        for cls, color, sign in ((0, "#5b9bd5", -1), (1, "#d9534f", +1)):
            vals = e[y == cls]
            if len(vals) < 2:
                continue
            d = _kde(np.clip(vals, 0, ymax), grid)
            d = d / d.max() * 0.4 if d.max() > 0 else d
            ax.fill_betweenx(grid, i, i + sign * d, color=color, alpha=0.65, lw=0)
            ax.plot(i, np.median(vals), "o", color="white", mec="black", ms=5, zorder=5)
    ax.axhline(np.mean([scores[s]["threshold"] for s in sids]), color="orange", ls="--", lw=1.5, label="Threshold θ")
    ax.set_xticks(range(len(sids))); ax.set_xticklabels([STATION_LABELS[s].replace(" (", "\n(") for s in sids], fontweight="bold")
    ax.set_ylim(0, ymax); ax.set_ylabel("Reconstruction Error")
    ax.set_title("(a) Reconstruction Error Distribution by Class")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#5b9bd5", alpha=0.65, label="Normal"), Patch(color="#d9534f", alpha=0.65, label="Anomalous"),
                       plt.Line2D([], [], color="orange", ls="--", label="Threshold θ")], loc="upper right")

    # (b) ROC ----------------------------------------------------------------
    ax = axes[0, 1]
    for sid in sids:
        e, y = scores[sid]["test"], scores[sid]["labels"]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, thr = roc_curve(y, e)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=STATION_COLORS[sid], lw=1.8, label=f"{C.STATIONS[sid]['short']} (AUC = {a:.3f})")
        j = np.argmin(np.abs(thr - scores[sid]["threshold"]))
        ax.plot(fpr[j], tpr[j], "D", color=STATION_COLORS[sid], ms=6)
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"(b) {detector} ROC Curves Across Stations"); ax.legend(loc="lower right", fontsize=8)

    # (c) log-scale separation, primary station ------------------------------
    ax = axes[1, 0]
    if primary in scores:
        e, y, th = scores[primary]["test"], scores[primary]["labels"], scores[primary]["threshold"]
        le = np.log10(np.clip(e, 1e-8, None))
        grid = np.linspace(le.min() - 0.3, le.max() + 0.3, 400)
        for cls, color, lab in ((0, "#4a86c8", "Normal"), (1, "#c0392b", "Anomalous")):
            v = le[y == cls]
            if len(v) < 2:
                continue
            d = _kde(v, grid); d = d / d.max() if d.max() > 0 else d
            ax.plot(grid, d, color=color, lw=1.6, label=f"{lab} (n={len(v)})")
            ax.fill_between(grid, d, color=color, alpha=0.25)
        ax.axvline(np.log10(th), color="orange", ls="--", lw=1.8, label="Threshold θ")
        ax.axvspan(grid.min(), np.log10(th), color="#4a86c8", alpha=0.06)
        ax.axvspan(np.log10(th), grid.max(), color="#c0392b", alpha=0.06)
        ax.set_xlabel("log₁₀(Reconstruction Error)"); ax.set_ylabel("Normalised Density")
        ax.set_title(f"(c) Log-Scale Error Separation — {C.STATIONS[primary]['name'].split(',')[0]}")
        ax.legend(loc="upper left", fontsize=8)

    # (d) threshold sensitivity ----------------------------------------------
    ax = axes[1, 1]
    for sid in sids:
        e, y, th = scores[sid]["test"], scores[sid]["labels"], scores[sid]["threshold"]
        if y.sum() == 0:
            continue
        grid, f1 = threshold_sweep(e, y)
        ax.plot(np.log10(grid), f1, color=STATION_COLORS[sid], lw=1.8, label=C.STATIONS[sid]["short"])
        ax.axvline(np.log10(th), color=STATION_COLORS[sid], ls=":", lw=1, alpha=0.7)
        j = np.argmin(np.abs(grid - th)); ax.plot(np.log10(th), f1[j], "D", color=STATION_COLORS[sid], ms=6)
    ax.set_xlabel("log₁₀(Threshold θ)"); ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1.05)
    ax.set_title("(d) Threshold Sensitivity Analysis"); ax.legend(ncol=2, fontsize=8, loc="lower left")

    fig.tight_layout()
    return _save(fig, "fig05_reconstruction_error", out_dir)


# ===========================================================================
# Fig. 6 — multivariate time series with detected anomalies (Clackamas)
# ===========================================================================
def fig06_timeseries(sd_like: dict, scores: np.ndarray, threshold: float, out_dir: Path = C.FIGURES_DIR,
                     title: Optional[str] = None, fixed_alpha: float = C.FIXED_ALPHA) -> list[Path]:
    """
    sd_like: {"columns", "test_series" (n_steps, 5) standardised, "mean", "std", "name"}
    scores:  per-window PatchTST error on the test partition.
    The x-axis is the flattened test time-step index ("Window Index" in the paper).
    """
    cols = sd_like["columns"]; series = sd_like["test_series"] * sd_like["std"] + sd_like["mean"]
    n_steps = series.shape[0]
    starts = C.WINDOW_STRIDE * np.arange(len(scores))
    flagged = np.where(scores > threshold)[0]

    palette = ["#1f77b4", "#20b2aa", "#7b68ee", "#f28e2b", "#8b5a2b"]
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(cols) + [1.1]})
    x = np.arange(n_steps)
    for i, (ax, c) in enumerate(zip(axes[:-1], cols)):
        ax.plot(x, series[:, i], color=palette[i % len(palette)], lw=0.6)
        disp, unit = C.PARAM_DISPLAY.get(c, (c, ""))
        ax.set_ylabel(f"{disp} ({unit})" if unit else disp)
        for w in flagged:
            ax.axvspan(starts[w], starts[w] + C.WINDOW_SIZE, color="#e74c3c", alpha=0.15, lw=0)
    ax = axes[-1]
    e_step = np.zeros(n_steps); cnt = np.zeros(n_steps)
    for w, s in enumerate(starts):                     # map window error to its span (mean of overlaps)
        e_step[s:s + C.WINDOW_SIZE] += scores[w]; cnt[s:s + C.WINDOW_SIZE] += 1
    e_step = np.divide(e_step, np.maximum(cnt, 1))
    anom_mask = e_step > threshold
    ax.fill_between(x, 0, e_step, where=~anom_mask, color="#7fd1c8", alpha=0.7, lw=0, label="Normal")
    ax.fill_between(x, 0, e_step, where=anom_mask, color="#e07b6b", alpha=0.9, lw=0, label="Anomalous")
    ax.axhline(threshold, color="#c0392b", ls="--", lw=1.6, label=f"Threshold (α = {fixed_alpha})")
    ax.set_ylabel("Recon. Error"); ax.set_xlabel("Window Index"); ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title or f"Water Quality Anomaly Detection — {sd_like['name']}", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save(fig, "fig06_timeseries", out_dir)


# ===========================================================================
# Fig. 7 — explanation quality across stations
# ===========================================================================
def fig07_explanation_quality(eval_df: pd.DataFrame, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    from .evaluation import DIMS, DIM_LABELS, table4_explanation_quality
    df = eval_df[eval_df["valid"]]
    t4 = table4_explanation_quality(df)
    st = t4[t4["station_id"] != "all"].set_index("station_id")
    grand = t4[t4["station_id"] == "all"].iloc[0]
    sids = [s for s in C.STATION_ORDER if s in st.index]
    lbl = ["Completeness", "Reg. Accuracy", "Actionability", "Overall"]
    M = np.array([[st.loc[s, f"{DIM_LABELS[d]}_mean"] for d in DIMS] for s in sids])
    S = np.array([[st.loc[s, f"{DIM_LABELS[d]}_sd"] for d in DIMS] for s in sids])

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # (a) heatmap
    ax = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("q", ["#b2182b", "#fddbc7", "#f7f7f7", "#d9f0a3", "#1a9850", "#00441b"])
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(sids)):
        for j in range(len(DIMS)):
            dark = M[i, j] > 0.92
            ax.text(j, i - 0.12, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if dark else "black")
            ax.text(j, i + 0.25, f"± {S[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="#dddddd" if dark else "#333")
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(lbl, fontweight="bold", fontsize=8.5)
    ax.set_yticks(range(len(sids))); ax.set_yticklabels([STATION_LABELS[s].replace(" (", "\n(") for s in sids], fontweight="bold")
    ax.grid(False); plt.colorbar(im, ax=ax, label="Quality Score", shrink=0.85)
    ax.set_title(f"(a) Explanation Quality Heatmap (n = {int(st['n'].iloc[0])} per station)")

    # (b) radar
    ax = fig.add_subplot(gs[0, 1], polar=True)
    ang = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist(); ang += ang[:1]
    for i, s in enumerate(sids):
        v = M[i].tolist(); v += v[:1]
        ax.plot(ang, v, "o-", lw=1.6, ms=4, color=STATION_COLORS[s], label=STATION_LABELS[s])
        ax.fill(ang, v, alpha=0.05, color=STATION_COLORS[s])
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(["Completeness", "Regulatory\nAccuracy", "Actionability", "Overall\nQuality"], fontweight="bold")
    ax.set_ylim(0.5, 1.0); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8)
    ax.set_title("(b) Quality Profile per Station", pad=18); ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.2, -0.05))

    # (c) grouped bars
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(DIMS)); w = 0.2
    for i, s in enumerate(sids):
        b = ax.bar(x + (i - 1.5) * w, M[i], w, yerr=S[i], capsize=2, color=STATION_COLORS[s], label=STATION_LABELS[s],
                   error_kw={"lw": 0.8, "alpha": 0.7})
        for bb, v in zip(b, M[i]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color=STATION_COLORS[s], fontweight="bold")
    ax.axhline(0.8, color="grey", ls="--", lw=0.8, alpha=0.6); ax.text(3.45, 0.805, "Good", fontsize=7, color="grey", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontweight="bold"); ax.set_ylim(0, 1.12); ax.set_ylabel("Quality Score")
    ax.set_title(f"(c) Per-Dimension Scores with Standard Deviation (n = {int(st['n'].iloc[0])})"); ax.legend(ncol=2, fontsize=7, loc="lower left")

    # (d) cross-station mean
    ax = fig.add_subplot(gs[1, 1])
    means = [grand[f"{DIM_LABELS[d]}_mean"] for d in DIMS]; sds = [grand[f"{DIM_LABELS[d]}_sd"] for d in DIMS]
    cols = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]
    y = np.arange(len(DIMS))[::-1]
    ax.barh(y, means, xerr=sds, color=cols, capsize=4, height=0.6, error_kw={"lw": 1.2})
    for yi, m, s, c in zip(y, means, sds, cols):
        ax.text(m + s + 0.01, yi, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9.5, fontweight="bold", color=c)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontweight="bold"); ax.set_xlim(0, 1.25); ax.set_xlabel("Cross-Station Mean Score")
    ax.set_title(f"(d) Cross-Station Mean ± SD (N = {int(grand['n'])})")
    return _save(fig, "fig07_explanation_quality", out_dir)


# ===========================================================================
# Fig. 8 — Llama-3-8B vs Mistral-7B
# ===========================================================================
def fig08_llm_comparison(summary: pd.DataFrame, paired: pd.DataFrame, out_dir: Path = C.FIGURES_DIR,
                         primary: str = C.LLM_PRIMARY, other: str = C.LLM_COMPARISON) -> list[Path]:
    from .evaluation import DIMS, DIM_LABELS
    lbl = ["Completeness", "Reg. Accuracy", "Actionability", "Overall"]
    names = {primary: "Llama-3-8B", other: "Mistral-7B"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    x = np.arange(4); w = 0.38
    for k, (m, color, off) in enumerate(((primary, "#2f6fd6", -w / 2), (other, "#e03c31", +w / 2))):
        row = summary[summary["LLM"] == m].iloc[0]
        mu = [row[f"{DIM_LABELS[d]}_mean"] for d in DIMS]; sd = [row[f"{DIM_LABELS[d]}_sd"] for d in DIMS]
        b = ax.bar(x + off, mu, w, yerr=sd, capsize=3, color=color, label=names.get(m, m), error_kw={"lw": 1})
        for bb, v in zip(b, mu):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=color, fontweight="bold")
    n = int(summary["n"].iloc[0])
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontweight="bold"); ax.set_ylim(0, 1.35); ax.set_ylabel("Score")
    ax.set_title(f"(a) Mean Quality Comparison (n = {n})"); ax.legend(loc="upper left")

    ax = axes[1]
    data = [paired[d].values for d in DIMS[1:]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="darkred", markersize=7),
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], ["#f7c873", "#6fe3c4", "#6fb6d6"]):
        patch.set_facecolor(color); patch.set_alpha(0.9)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(lbl[1:], fontweight="bold")
    ax.set_ylabel(f"Δ Score ({names.get(primary,'A')} − {names.get(other,'B').split('-')[0]})")
    ax.set_title("(b) Paired Score Differences")
    ax.text(1.02, 0.5, "mean", color="red", transform=ax.transAxes, fontsize=8, va="center")
    fig.tight_layout()
    return _save(fig, "fig08_llm_comparison", out_dir)


# ===========================================================================
# Fig. 9 — confusion matrices
# ===========================================================================
def fig09_confusion_matrices(table3: pd.DataFrame, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    fig, axes = plt.subplots(4, 4, figsize=(13, 12.5))
    cmap = plt.cm.Blues
    for r, model in enumerate(C.MODEL_ORDER):
        for c, sid in enumerate(C.STATION_ORDER):
            ax = axes[r, c]
            row = table3[(table3["station_id"].astype(str) == sid) & (table3["Model"] == model)]
            ax.grid(False)
            if row.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes); ax.axis("off"); continue
            tn, fp, fn, tp = (int(row[k].iloc[0]) for k in ("TN", "FP", "FN", "TP"))
            cm = np.array([[tn, fp], [fn, tp]]); tot = cm.sum()
            ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.15)
            for i in range(2):
                for j in range(2):
                    col = "white" if cm[i, j] > cm.max() * 0.55 else "black"
                    ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / tot * 100:.1f}%)", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=col)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Anomaly"] if r == 3 else ["", ""], fontsize=8)
            ax.set_yticklabels(["Normal", "Anomaly"] if c == 0 else ["", ""], fontsize=8)
            if r == 0:
                ax.set_title(C.STATIONS[sid]["short"], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"({'abcd'[r]})\n{model}", fontsize=10, fontweight="bold")
    fig.suptitle("Confusion Matrices Across All Models and Stations", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save(fig, "fig09_confusion_matrices", out_dir)


# ===========================================================================
# Fig. 10 — best model per station
# ===========================================================================
def fig10_best_model(table3: pd.DataFrame, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    names, f1s, colors, labels = [], [], [], []
    for sid in C.STATION_ORDER:
        s = table3[table3["station_id"].astype(str) == sid]
        if s.empty:
            continue
        best = s.loc[s["F1"].idxmax()]
        names.append(C.STATIONS[sid]["short"]); f1s.append(best["F1"]); labels.append(best["Model"])
        colors.append("#0fa37f" if best["Model"] == "LSTM-AE" else "#e74c3c" if best["Model"] == "OC-SVM"
                      else C.MODEL_COLORS.get(best["Model"], "#888"))
    b = ax.bar(names, f1s, color=colors, width=0.55)
    for bb, f, l in zip(b, f1s, labels):
        ax.text(bb.get_x() + bb.get_width() / 2, f + 0.015, f"{l}\nF1 = {f:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.95); ax.set_ylabel("Best F1-Score"); ax.set_title("Best Performing Model Per Station", fontsize=14)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig10_best_model", out_dir)
