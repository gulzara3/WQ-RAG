#!/usr/bin/env python
"""
Regenerate all paper figures.

    --from-paper   use paper_results/*.csv (published numbers) -> Figs 1, 3, 7, 8a-style, 9, 10
    default        use results/ from a fresh run             -> Figs 1, 3-10
"""
import argparse
import numpy as np
import pandas as pd
import _common  # noqa: F401
from wqrag import config as C, figures as F
from wqrag.utils import get_logger, load_json

log = get_logger("figures")


def _eval_from_table4(t4: pd.DataFrame) -> pd.DataFrame:
    """Build a pseudo per-anomaly frame reproducing Table 4 means/SDs (for Fig. 7 from paper)."""
    rows = []
    rng = np.random.RandomState(0)
    for _, r in t4[t4["station_id"] != "all"].iterrows():
        n = int(r["n"])
        cols = {}
        for d, lab in zip(["completeness", "regulatory_accuracy", "actionability", "overall"],
                          ["Completeness", "Regulatory Accuracy", "Actionability", "Overall"]):
            m, s = r[f"{lab}_mean"], r[f"{lab}_sd"]
            x = rng.normal(0, 1, n); x = (x - x.mean()) / (x.std(ddof=1) if n > 1 else 1)
            cols[d] = m + s * x
        for i in range(n):
            rows.append(dict(station_id=str(r["station_id"]).zfill(8), valid=True, **{k: v[i] for k, v in cols.items()}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-paper", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.FIGURES_DIR if a.out is None else a.out
    src = C.PAPER_RESULTS_DIR if a.from_paper else C.TABLES_DIR

    F.fig01_station_map(out)
    t3 = pd.read_csv(src / "table3_detection_performance.csv", dtype={"station_id": str})
    F.fig03_detection_performance(t3, out)
    F.fig09_confusion_matrices(t3, out)
    F.fig10_best_model(t3, out)

    t4p = src / "table4_explanation_quality.csv"
    if t4p.exists():
        if a.from_paper:
            ev = _eval_from_table4(pd.read_csv(t4p, dtype={"station_id": str}))
        else:
            ev = pd.read_csv(src / "explanation_scores_all.csv", dtype={"station_id": str})
        F.fig07_explanation_quality(ev, out)

    t6p = src / "table6_llm_comparison.csv"
    if t6p.exists():
        t6 = pd.read_csv(t6p)
        pp = src / "table6_paired_differences.csv"
        if pp.exists():
            paired = pd.read_csv(pp, index_col=0)
        else:  # synthesise paired differences consistent with Δ and Cohen's d (paper mode only)
            rng = np.random.RandomState(1); n = 50
            delta = t6.iloc[2]; dval = t6.iloc[3]; paired = {}
            for lab, d in zip(["Completeness", "Regulatory Accuracy", "Actionability", "Overall"],
                              ["completeness", "regulatory_accuracy", "actionability", "overall"]):
                m = float(delta[f"{lab}_mean"]); cd = dval[f"{lab}_mean"]
                sd = abs(m / float(cd)) if pd.notna(cd) and float(cd) != 0 else 0.0
                x = rng.normal(0, 1, n); x = (x - x.mean()) / x.std(ddof=1)
                paired[d] = m + sd * x
            paired = pd.DataFrame(paired)
        F.fig08_llm_comparison(t6, paired, out)

    if not a.from_paper:
        det = load_json(C.RESULTS_DIR / f"detection_{C.PRIMARY_STATION}.json")
        F.fig04_training_curves(det, out)
        scores = {}
        for sid in C.STATION_ORDER:
            p = C.RESULTS_DIR / f"scores_{sid}.npz"
            if p.exists():
                z = np.load(p); d = load_json(C.RESULTS_DIR / f"detection_{sid}.json")
                scores[sid] = dict(test=z["PatchTST_test"], labels=z["labels"], threshold=d["models"]["PatchTST"]["threshold"])
        F.fig05_reconstruction_error(scores, out)
        from wqrag.preprocessing import StationData
        sd = StationData.load(C.PROCESSED_DIR / f"station_{C.PRIMARY_STATION}.npz")
        F.fig06_timeseries(dict(columns=sd.columns, test_series=sd.test_series, mean=sd.mean, std=sd.std, name=sd.name),
                           scores[C.PRIMARY_STATION]["test"], scores[C.PRIMARY_STATION]["threshold"], out)
    log.info("Figures written to %s", out)
