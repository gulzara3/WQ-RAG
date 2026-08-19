import numpy as np
import pandas as pd
from wqrag import config as C
from wqrag.evaluation import (cohens_d_paired, effect_size_label, evaluate_explanation, evaluate_many,
                              extreme_events, table4_explanation_quality, table5_ablation, table7_extreme_events)
from wqrag.explainer import build_retrieval_query, format_anomaly_context

GOOD = """1. PROBABLE ROOT CAUSE: storm-driven sediment mobilisation caused by post-wildfire runoff.
2. SEVERITY ASSESSMENT: HIGH. Turbidity 180 FNU exceeds the EPA NPDWR 1 NTU limit.
3. REGULATORY CONTEXT: EPA Secondary Standards pH 6.5-8.5; WHO guideline value; MCL not applicable.
4. RECOMMENDED ACTIONS: Immediately notify the utility operator; sample within 24 hours; monitor.
5. CONFIDENCE LEVEL: 8/10."""


def test_full_marks_explanation():
    e = evaluate_explanation(GOOD)
    assert e["completeness"] == 1.0 and e["regulatory_accuracy"] == 1.0 and e["actionability"] == 1.0
    assert evaluate_explanation("Error: timeout")["valid"] is False


def test_cohens_d():
    a = np.array([0.8, 0.9, 0.7, 1.0]); b = np.array([0.6, 0.7, 0.6, 0.8])
    d = cohens_d_paired(a, b)
    assert d > 0 and effect_size_label(d) == "large"


def test_tables_build():
    recs = [dict(station_id=f"USGS-{s}", station_name=C.STATIONS[s]["name"], rank=i, severity_score=0.5,
                 explanation=GOOD, llm_model="x") for s in C.STATION_ORDER for i in range(3)]
    ev = evaluate_many(recs)
    t4 = table4_explanation_quality(ev)
    assert len(t4) == 5 and t4.iloc[-1]["station_id"] == "all"
    t5 = table5_ablation({c: recs[:2] for c in C.ABLATION_CONFIGS})
    assert list(t5["Configuration"]) == C.ABLATION_CONFIGS


def test_query_and_context():
    devs = {"Temperature_C": 0.2, "Conductivity_uScm": 4.1, "DO_mgL": -2.5, "pH": 0.1, "Turbidity_FNU": 9.0}
    q = build_retrieval_query(devs)
    assert q.startswith("water quality anomaly: Turbidity_FNU above normal, Conductivity_uScm above normal, DO_mgL below normal")
    a = dict(station_id="USGS-1", station_name="X", severity_score=0.7, parameters={k: 1.0 for k in devs}, deviations=devs)
    assert "⚠️" in format_anomaly_context(a)


def test_extreme_events_table():
    from wqrag.preprocessing import StationData
    te = np.zeros((20, 96, 5), dtype=np.float32); te[3, 5, 4] = 6.0; te[7, 0, 1] = -4.5
    sd = StationData("14211010", "Clackamas River, OR", C.PARAM_ORDER, np.zeros(5), np.ones(5),
                     te[:2], te[:2], te, np.zeros(2), np.zeros(2), np.zeros(20),
                     pd.date_range("2024-01-01", periods=96 * 20, freq="15min"), np.zeros((96 * 20, 5)))
    ev = extreme_events(sd)
    assert len(ev) == 2 and set(ev["dominant_parameter"]) == {"Turbidity_FNU", "Conductivity_uScm"}
    t7 = table7_extreme_events({"14211010": ev})
    assert t7.iloc[-1]["Events"] == 2 and abs(t7.iloc[0]["Max |z|"] - 6.0) < 1e-6
