#!/usr/bin/env python
"""Score all explanations and write Tables 4-7 to results/tables/."""
import _common  # noqa: F401
import pandas as pd
from wqrag import config as C
from wqrag.evaluation import (evaluate_many, extreme_events, parameter_breakdown, table4_explanation_quality,
                              table5_ablation, table6_llm_comparison, table7_extreme_events)
from wqrag.preprocessing import prepare_all
from wqrag.utils import get_logger, load_json

log = get_logger("evaluate")

if __name__ == "__main__":
    C.ensure_dirs()
    T = C.TABLES_DIR
    # Table 4 --------------------------------------------------------------
    recs = []
    for sid in C.STATION_ORDER:
        p = C.EXPLANATIONS_DIR / f"explanations_{sid}.json"
        if p.exists():
            recs += load_json(p)
    if recs:
        ev = evaluate_many(recs)
        ev.to_csv(T / "explanation_scores_all.csv", index=False)
        t4 = table4_explanation_quality(ev)
        t4.to_csv(T / "table4_explanation_quality.csv", index=False)
        log.info("Table 4\n%s", t4.round(3).to_string(index=False))
    # Table 5 --------------------------------------------------------------
    p = C.EXPLANATIONS_DIR / f"ablation_{C.PRIMARY_STATION}.json"
    if p.exists():
        t5 = table5_ablation(load_json(p))
        t5.to_csv(T / "table5_ablation.csv", index=False)
        log.info("Table 5\n%s", t5.round(2).to_string(index=False))
    # Table 6 --------------------------------------------------------------
    p = C.EXPLANATIONS_DIR / f"llm_comparison_{C.PRIMARY_STATION}.json"
    if p.exists():
        t6, paired = table6_llm_comparison(load_json(p))
        t6.to_csv(T / "table6_llm_comparison.csv", index=False)
        paired.to_csv(T / "table6_paired_differences.csv")
        log.info("Table 6\n%s\nwins/ties/losses = %s", t6.round(3).to_string(index=False), paired.attrs)
    # Table 7 --------------------------------------------------------------
    data = prepare_all()
    events = {sid: extreme_events(sd) for sid, sd in data.items()}
    for sid, ev in events.items():
        ev.to_csv(T / f"extreme_events_{sid}.csv", index=False)
    t7 = table7_extreme_events(events)
    t7.to_csv(T / "table7_extreme_events.csv", index=False)
    log.info("Table 7\n%s\nparameter breakdown:\n%s", t7.round(1).to_string(index=False), parameter_breakdown(events))
