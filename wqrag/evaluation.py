"""
Evaluation framework (Section 2.4.2).

Explanation quality (automated text analysis, N = 200):
  * Completeness         fraction of the 5 required sections present
  * Regulatory accuracy  fraction of 5 checks satisfied: EPA mention, WHO mention,
                         numeric threshold with units, MCL/limit terminology,
                         named framework / specific guideline document
  * Actionability        fraction of 4 checks satisfied: immediate action,
                         specific operational instruction, follow-up timeframe,
                         reference to responsible authority/operator
  * Overall              mean of the three dimensions

Also: paired Cohen's d for the multi-LLM comparison (Table 6, Fig. 8),
extreme-event catalogue |z| > 4 sigma (Table 7), and builders for Tables 4-7.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .preprocessing import StationData

# ---------------------------------------------------------------------------
# Per-explanation scoring
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = {
    "root_cause": r"(?i)(root\s*cause|probable\s*cause|caused?\s*by|likely\s*cause)",
    "severity": r"(?i)(severity|critical|\bhigh\b|\bmedium\b|\blow\b|danger)",
    "regulatory": r"(?i)(EPA|WHO|MCL|regulation|standard|guideline|threshold|limit)",
    "actions": r"(?i)(recommend|action|should|immediately|investigate|monitor|check)",
    "confidence": r"(?i)(confidence|certain|uncertain|\d+\s*/\s*10|\d+\s*out\s*of)",
}
_REGULATORY_CHECKS = {
    "mentions_epa": r"\bEPA\b",
    "mentions_who": r"\bWHO\b",
    "numeric_threshold_with_units": r"\d+\.?\d*\s*(mg/L|mg/l|NTU|FNU|µS|uS|μS|°C|degrees)",
    "mentions_mcl_or_limit": r"(?i)(\bMCL\b|maximum\s*contaminant|permissible\s*limit|guideline\s*value)",
    "names_specific_standard": r"(?i)(NPDWR|secondary\s*(drinking\s*water\s*)?standard|"
                               r"drinking[- ]water\s*regulation|guidelines\s*for\s*drinking|"
                               r"basin\s*plan|water\s*quality\s*standards|"
                               r"clean\s*water\s*act|safe\s*drinking\s*water\s*act|\bSDWA\b)",
}
_ACTIONABILITY_CHECKS = {
    "immediate_action": r"(?i)(immediate|right\s*away|\bnow\b|urgent|\bfirst\b)",
    "specific_instruction": r"(?i)(check|verify|sample|test|inspect|measure|notify|contact|calibrat|flush)",
    "followup_timeframe": r"(?i)(24\s*hour|48\s*hour|follow.?up|monitor|recheck|within\s*\d+)",
    "responsible_party": r"(?i)(operator|utility|personnel|staff|treatment\s*plant|agency|authority|department)",
}


def _score(text: str, patterns: Dict[str, str]) -> Dict[str, bool]:
    return {k: bool(re.search(p, text)) for k, p in patterns.items()}


def evaluate_explanation(text: str) -> dict:
    if not text or text.startswith("Error"):
        return dict(valid=False, completeness=np.nan, regulatory_accuracy=np.nan,
                    actionability=np.nan, overall=np.nan, word_count=0)
    s = _score(text, _SECTION_PATTERNS)
    r = _score(text, _REGULATORY_CHECKS)
    a = _score(text, _ACTIONABILITY_CHECKS)
    comp = sum(s.values()) / len(s)
    reg = sum(r.values()) / len(r)
    act = sum(a.values()) / len(a)
    return dict(valid=True, completeness=comp, regulatory_accuracy=reg, actionability=act,
                overall=float(np.mean([comp, reg, act])), word_count=len(text.split()),
                sections=s, regulatory_checks=r, actionability_checks=a)


def evaluate_many(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        e = evaluate_explanation(rec.get("explanation", ""))
        rows.append(dict(
            station_id=rec.get("station_id", "").replace("USGS-", ""),
            station=rec.get("station_name", ""), rank=rec.get("rank"),
            severity_score=rec.get("severity_score"), llm_model=rec.get("llm_model"),
            completeness=e["completeness"], regulatory_accuracy=e["regulatory_accuracy"],
            actionability=e["actionability"], overall=e["overall"], word_count=e["word_count"],
            valid=e["valid"],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 4, 5, 6
# ---------------------------------------------------------------------------
DIMS = ["completeness", "regulatory_accuracy", "actionability", "overall"]
DIM_LABELS = {"completeness": "Completeness", "regulatory_accuracy": "Regulatory Accuracy",
              "actionability": "Actionability", "overall": "Overall"}


def table4_explanation_quality(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Per-station mean ± SD and the cross-station mean ± SD-of-station-means (Table 4)."""
    df = eval_df[eval_df["valid"]].copy()
    rows = []
    for sid in C.STATION_ORDER:
        s = df[df["station_id"] == sid]
        if s.empty:
            continue
        row = {"Station": C.STATIONS[sid]["name"], "station_id": sid, "n": len(s)}
        for d in DIMS:
            row[f"{DIM_LABELS[d]}_mean"] = s[d].mean()
            row[f"{DIM_LABELS[d]}_sd"] = s[d].std(ddof=1)
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        grand = {"Station": "Cross-station mean", "station_id": "all", "n": int(t["n"].sum())}
        for d in DIMS:
            grand[f"{DIM_LABELS[d]}_mean"] = t[f"{DIM_LABELS[d]}_mean"].mean()
            grand[f"{DIM_LABELS[d]}_sd"] = t[f"{DIM_LABELS[d]}_mean"].std(ddof=1)
        t = pd.concat([t, pd.DataFrame([grand])], ignore_index=True)
    return t


def table5_ablation(ablation: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for cfg in C.ABLATION_CONFIGS:
        recs = ablation.get(cfg, [])
        e = evaluate_many(recs)
        e = e[e["valid"]]
        if e.empty:
            continue
        rows.append({"Configuration": cfg, "Completeness": e["completeness"].mean(),
                     "Regulatory Accuracy": e["regulatory_accuracy"].mean(),
                     "Actionability": e["actionability"].mean(), "Overall": e["overall"].mean(),
                     "n": len(e)})
    return pd.DataFrame(rows)


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Paired-samples Cohen's d = mean(diff) / sd(diff)  (Cohen, 1988)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def effect_size_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def table6_llm_comparison(comparison: Dict[str, List[dict]],
                          primary: str = C.LLM_PRIMARY, other: str = C.LLM_COMPARISON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary table, paired per-anomaly differences)."""
    a = evaluate_many(comparison[primary]).set_index("rank")
    b = evaluate_many(comparison[other]).set_index("rank")
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    summary = []
    for name, df in ((primary, a), (other, b)):
        summary.append({"LLM": name, **{DIM_LABELS[d] + "_mean": df[d].mean() for d in DIMS},
                        **{DIM_LABELS[d] + "_sd": df[d].std(ddof=1) for d in DIMS},
                        "words_mean": df["word_count"].mean(), "words_sd": df["word_count"].std(ddof=1), "n": len(df)})
    delta = {"LLM": f"Δ ({primary} − {other})", "n": len(idx)}
    dval = {"LLM": "Cohen's d", "n": len(idx)}
    for d in DIMS:
        delta[DIM_LABELS[d] + "_mean"] = a[d].mean() - b[d].mean()
        dd = cohens_d_paired(a[d].values, b[d].values)
        dval[DIM_LABELS[d] + "_mean"] = dd
    summary.extend([delta, dval])
    paired = pd.DataFrame({d: a[d].values - b[d].values for d in DIMS}, index=idx)
    wins = int((paired["overall"] > 0).sum()); ties = int((paired["overall"] == 0).sum())
    losses = int((paired["overall"] < 0).sum())
    paired.attrs.update(wins=wins, ties=ties, losses=losses)
    return pd.DataFrame(summary), paired


# ---------------------------------------------------------------------------
# Table 7 — extreme events |z| > 4 sigma (independent physical validation)
# ---------------------------------------------------------------------------
def extreme_events(sd: StationData, sigma: float = C.EXTREME_EVENT_SIGMA) -> pd.DataFrame:
    """One row per test window whose max |z| over any parameter/step exceeds sigma."""
    if len(sd.test) == 0:
        return pd.DataFrame()
    absz = np.abs(sd.test)                         # (n, 96, 5)
    per_win_max = absz.max(axis=(1, 2))
    mask = per_win_max > sigma
    idx = np.where(mask)[0]
    starts = C.WINDOW_STRIDE * np.arange(len(sd.test))
    rows = []
    for i in idx:
        step, p = np.unravel_index(np.argmax(absz[i]), absz[i].shape)
        ts = sd.test_index[starts[i] + step] if len(sd.test_index) > starts[i] + step else None
        rows.append(dict(station_id=sd.station_id, station=sd.name, window_index=int(i),
                         timestamp=str(ts), dominant_parameter=sd.columns[p],
                         max_abs_z=float(absz[i, step, p]), signed_z=float(sd.test[i, step, p]),
                         value=float(sd.inverse_transform(sd.test[i, step])[p])))
    return pd.DataFrame(rows)


def table7_extreme_events(events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    all_z = []
    for sid in C.STATION_ORDER:
        ev = events.get(sid)
        if ev is None or ev.empty:
            continue
        counts = ev["dominant_parameter"].value_counts()
        top2 = ", ".join(f"{C.PARAM_DISPLAY[p][0]} ({n})" for p, n in counts.head(2).items())
        rows.append({"Station": C.STATIONS[sid]["name"], "Events": len(ev), "Dominant Parameter": top2,
                     "Max |z|": ev["max_abs_z"].max(), "Mean |z|": ev["max_abs_z"].mean(),
                     "Median |z|": ev["max_abs_z"].median()})
        all_z.append(ev["max_abs_z"].values)
    t = pd.DataFrame(rows)
    if all_z:
        z = np.concatenate(all_z)
        t = pd.concat([t, pd.DataFrame([{"Station": "Total", "Events": len(z), "Dominant Parameter": "",
                                         "Max |z|": z.max(), "Mean |z|": z.mean(), "Median |z|": np.median(z)}])],
                      ignore_index=True)
    return t


def parameter_breakdown(events: Dict[str, pd.DataFrame]) -> pd.Series:
    """Counts of dominant parameter across all stations (text of Section 3.8.1)."""
    frames = [e for e in events.values() if e is not None and not e.empty]
    if not frames:
        return pd.Series(dtype=int)
    return pd.concat(frames)["dominant_parameter"].value_counts()
