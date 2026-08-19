#!/usr/bin/env python
"""Generate the 50 RAG-grounded explanations per station (200 total; Table 4, Fig. 7)."""
import argparse
import numpy as np
import _common  # noqa: F401
from wqrag import config as C
from wqrag.explainer import check_ollama, extract_anomalies, generate_explanations, make_full_chain, save_explanations
from wqrag.knowledge_base import build_rag_backend
from wqrag.preprocessing import prepare_all
from wqrag.utils import get_logger, load_json

log = get_logger("explain")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", nargs="*", default=C.STATION_ORDER)
    ap.add_argument("--detector", default="PatchTST", help="which detector's scores rank the anomalies")
    ap.add_argument("--n", type=int, default=C.N_EXPLANATIONS_PER_STATION)
    ap.add_argument("--model", default=C.LLM_PRIMARY)
    a = ap.parse_args()
    C.ensure_dirs()
    if not check_ollama([a.model]):
        raise SystemExit(1)
    _, retriever = build_rag_backend()
    chain = make_full_chain(retriever, a.model)
    data = prepare_all(a.station)
    for sid, sd in data.items():
        det = load_json(C.RESULTS_DIR / f"detection_{sid}.json")
        sc = np.load(C.RESULTS_DIR / f"scores_{sid}.npz")
        scores = sc[f"{a.detector}_test"]
        theta = det["models"][a.detector]["threshold"]
        anomalies = extract_anomalies(sd, scores, theta, top_n=a.n, detector=a.detector)
        log.info("%s: %d anomalies selected", sd.name, len(anomalies))
        recs = generate_explanations(anomalies, chain, model=a.model)
        save_explanations(recs, C.EXPLANATIONS_DIR / f"explanations_{sid}.json")
