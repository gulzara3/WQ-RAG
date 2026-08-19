#!/usr/bin/env python
"""Diff a fresh run (results/tables) against the published tables (paper_results)."""
import pandas as pd
import _common  # noqa: F401
from wqrag import config as C

if __name__ == "__main__":
    for name, keys, cols in [
        ("table3_detection_performance.csv", ["station_id", "Model"], ["Precision", "Recall", "F1", "CSI", "AUC"]),
        ("table5_ablation.csv", ["Configuration"], ["Completeness", "Regulatory Accuracy", "Actionability", "Overall"]),
    ]:
        a = C.PAPER_RESULTS_DIR / name; b = C.TABLES_DIR / name
        if not b.exists():
            print(f"[skip] {name} (no fresh run)"); continue
        pa = pd.read_csv(a, dtype={"station_id": str}); pb = pd.read_csv(b, dtype={"station_id": str})
        m = pa.merge(pb, on=keys, suffixes=("_paper", "_run"))
        for c in cols:
            if f"{c}_run" in m:
                m[f"{c}_diff"] = m[f"{c}_run"] - m[f"{c}_paper"]
        print(f"\n== {name} ==")
        print(m[keys + [x for x in m.columns if x.endswith("_diff")]].round(3).to_string(index=False))
