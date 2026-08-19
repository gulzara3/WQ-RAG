#!/usr/bin/env python
"""Llama-3-8B vs Mistral-7B on the same 50 Clackamas anomalies (Table 6, Fig. 8)."""
import argparse
import _common  # noqa: F401
from wqrag import config as C
from wqrag.explainer import check_ollama, run_llm_comparison, save_explanations
from wqrag.knowledge_base import build_rag_backend
from wqrag.utils import load_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", default=C.PRIMARY_STATION)
    ap.add_argument("--n", type=int, default=C.N_LLM_COMPARISON)
    a = ap.parse_args()
    if not check_ollama():
        raise SystemExit(1)
    anomalies = load_json(C.EXPLANATIONS_DIR / f"explanations_{a.station}.json")
    _, retriever = build_rag_backend()
    res = run_llm_comparison(anomalies, retriever, n=a.n)
    save_explanations(res, C.EXPLANATIONS_DIR / f"llm_comparison_{a.station}.json")
