#!/usr/bin/env python
"""Ablation study: 4 configurations on n=5 Clackamas anomalies (Table 5)."""
import argparse
import _common  # noqa: F401
from wqrag import config as C
from wqrag.explainer import check_ollama, run_ablation, save_explanations
from wqrag.knowledge_base import build_rag_backend
from wqrag.utils import load_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", default=C.PRIMARY_STATION)
    ap.add_argument("--n", type=int, default=C.N_ABLATION)
    a = ap.parse_args()
    if not check_ollama([C.LLM_PRIMARY]):
        raise SystemExit(1)
    anomalies = load_json(C.EXPLANATIONS_DIR / f"explanations_{a.station}.json")
    _, retriever = build_rag_backend()
    res = run_ablation(anomalies, retriever, n=a.n)
    save_explanations(res, C.EXPLANATIONS_DIR / f"ablation_{a.station}.json")
