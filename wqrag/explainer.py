"""
Stage IV — LLM explanation generation (Section 2.2.4; Fig. 2 Stage IV).

* Anomalies are taken from the detector's test-set scores (windows above the
  adaptive threshold), ranked by reconstruction error, top-50 per station.
* For each anomaly the peak time step gives parameter values (physical units),
  z-scores, a severity score in [0,1] = (e - theta)/(max e - theta), and a
  retrieval query built from the three most-deviated parameters + direction.
* Prompt: five mandatory sections (root cause, severity, regulatory context,
  recommended actions, confidence).  Zero-shot, temperature 0.1, 1,024 tokens.
* Primary model Llama-3-8B via Ollama; Mistral-7B under identical settings.
* Ablation (Clackamas, n = 5): Full Pipeline / LLM Only / LLM + RAG / LLM + Context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from . import config as C
from .preprocessing import StationData
from .utils import get_logger, save_json

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts (verbatim structure used for the 200 explanations)
# ---------------------------------------------------------------------------
FULL_PROMPT = """You are WQ-RAG, an expert water quality diagnostic AI assistant.
An anomaly has been detected by an automated monitoring system.
Your task is to diagnose the root cause and recommend corrective actions.

══════════════════════════════════════════════════════
ANOMALY CONTEXT (from sensor data):
══════════════════════════════════════════════════════
{anomaly_context}

══════════════════════════════════════════════════════
RELEVANT KNOWLEDGE (retrieved from regulatory standards and case studies):
══════════════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════════════
INSTRUCTIONS: Provide a structured diagnosis using EXACTLY this format:
══════════════════════════════════════════════════════

1. PROBABLE ROOT CAUSE:
[Identify the most likely cause of this anomaly based on the parameter deviations and retrieved knowledge. Be specific — name the physical, chemical, or operational process.]

2. SEVERITY ASSESSMENT:
[Rate as LOW / MEDIUM / HIGH / CRITICAL. Reference specific EPA or WHO threshold values from the retrieved documents.]

3. REGULATORY CONTEXT:
[Cite the specific regulations that apply. Include MCL values, guideline ranges, and which standard (EPA NPDWR, EPA Secondary, WHO, etc.) is relevant.]

4. RECOMMENDED ACTIONS:
[Provide 2-3 specific, actionable steps the operator should take immediately and within 24 hours.]

5. CONFIDENCE LEVEL:
[Rate your confidence 1-10 and explain what additional information would increase certainty.]

Be concise, factual, and reference the retrieved regulatory documents where possible."""

# Ablation prompts ----------------------------------------------------------
LLM_ONLY_PROMPT = """You are a water quality expert. An anomaly was detected:

{anomaly_context}

Provide a diagnosis with: 1) Probable root cause, 2) Severity assessment,
3) Regulatory context, 4) Recommended actions, 5) Confidence level."""

LLM_RAG_NO_CONTEXT_PROMPT = """You are a water quality expert. Using ONLY the retrieved knowledge below,
explain what could cause a general water quality anomaly.

Retrieved Knowledge:
{retrieved_context}

Provide: 1) Common causes, 2) Severity ranges, 3) Regulatory thresholds, 4) Actions."""

GENERIC_ANOMALY_CONTEXT = "A general water quality anomaly was detected at a monitoring station."


# ---------------------------------------------------------------------------
# Anomaly extraction from detector outputs
# ---------------------------------------------------------------------------
def build_retrieval_query(deviations: Dict[str, float], top_n: int = 3) -> str:
    top = sorted(deviations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return "water quality anomaly: " + ", ".join(
        f"{k} {'above' if v > 0 else 'below'} normal" for k, v in top)


def format_anomaly_context(a: dict) -> str:
    lines = [
        f"Station: {a['station_id']} — {a['station_name']}",
        f"Timestamp: {a.get('timestamp', 'unknown')}",
        f"Severity Score: {a['severity_score']:.3f}",
        "Anomaly Type: Reconstruction error exceeded threshold",
        "",
        "Parameter Values at Time of Anomaly:",
    ]
    for p, v in a["parameters"].items():
        z = a["deviations"][p]
        flag = "  ⚠️ ANOMALOUS" if abs(z) > 2 else ""
        lines.append(f"  {p}: {v:.3f}  (z-score: {z:+.2f}){flag}")
    return "\n".join(lines)


def extract_anomalies(sd: StationData, scores: np.ndarray, threshold: float,
                      top_n: int = C.N_EXPLANATIONS_PER_STATION, detector: str = "PatchTST") -> List[dict]:
    """Rank flagged test windows by reconstruction error and build anomaly dicts."""
    flagged = np.where(scores > threshold)[0]
    if len(flagged) == 0:
        return []
    order = flagged[np.argsort(scores[flagged])[::-1]][:top_n]
    max_score = float(scores.max())
    starts = C.WINDOW_STRIDE * np.arange(len(sd.test))
    out = []
    for rank, idx in enumerate(order, 1):
        w = sd.test[idx]                                   # (96, 5) standardised
        peak = int(np.argmax(np.abs(w).sum(axis=1)))
        z = w[peak].astype(float)
        phys = sd.inverse_transform(z)
        sev = float(np.clip((scores[idx] - threshold) / (max_score - threshold + 1e-8), 0, 1))
        ts = sd.test_index[starts[idx] + peak] if len(sd.test_index) > starts[idx] + peak else None
        devs = {c: float(z[i]) for i, c in enumerate(sd.columns)}
        a = dict(
            station_id=f"USGS-{sd.station_id}", station_name=sd.name, detector=detector,
            window_index=int(idx), peak_step=peak,
            timestamp=str(ts) if ts is not None else f"window {idx} step {peak}",
            severity_score=sev, reconstruction_error=float(scores[idx]), threshold=float(threshold),
            parameters={c: float(phys[i]) for i, c in enumerate(sd.columns)},
            deviations=devs, rank=rank,
        )
        a["retrieval_query"] = build_retrieval_query(devs)
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# LLM chains (Ollama via LangChain)
# ---------------------------------------------------------------------------
def _llm(model: str):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=C.LLM_TEMPERATURE, base_url=C.OLLAMA_BASE_URL,
                      num_predict=C.LLM_MAX_TOKENS)


def make_full_chain(retriever, model: str = C.LLM_PRIMARY) -> Callable[[dict], str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from .knowledge_base import format_docs
    chain = ({"retrieved_context": RunnableLambda(lambda x: x["query"]) | retriever | RunnableLambda(format_docs),
              "anomaly_context": RunnableLambda(lambda x: x["anomaly_context"])}
             | ChatPromptTemplate.from_template(FULL_PROMPT) | _llm(model) | StrOutputParser())
    return chain.invoke


def make_llm_only_chain(model: str = C.LLM_PRIMARY) -> Callable[[dict], str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    chain = ({"anomaly_context": RunnableLambda(lambda x: x["anomaly_context"])}
             | ChatPromptTemplate.from_template(LLM_ONLY_PROMPT) | _llm(model) | StrOutputParser())
    return chain.invoke


def make_rag_no_context_chain(retriever, model: str = C.LLM_PRIMARY) -> Callable[[dict], str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from .knowledge_base import format_docs
    chain = ({"retrieved_context": RunnableLambda(lambda x: x["query"]) | retriever | RunnableLambda(format_docs)}
             | ChatPromptTemplate.from_template(LLM_RAG_NO_CONTEXT_PROMPT) | _llm(model) | StrOutputParser())
    return chain.invoke


def check_ollama(models=(C.LLM_PRIMARY, C.LLM_COMPARISON)) -> bool:
    import requests
    try:
        r = requests.get(f"{C.OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        have = [m["name"] for m in r.json().get("models", [])]
        ok = True
        for m in models:
            found = any(m.split(":")[0] in h for h in have)
            log.info("  %s %s", "✓" if found else "✗", m)
            ok &= found
        if not ok:
            log.warning("Missing models — run: ollama pull <model>")
        return ok
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot reach Ollama at %s (%s). Start it with `ollama serve`.", C.OLLAMA_BASE_URL, exc)
        return False


# ---------------------------------------------------------------------------
# Generation routines
# ---------------------------------------------------------------------------
def _safe(call: Callable, payload: dict) -> str:
    try:
        return call(payload)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {str(exc)[:200]}"


def generate_explanations(anomalies: List[dict], full_chain: Callable, model: str = C.LLM_PRIMARY,
                          verbose: bool = True) -> List[dict]:
    out = []
    for a in anomalies:
        payload = {"query": a["retrieval_query"], "anomaly_context": format_anomaly_context(a)}
        text = _safe(full_chain, payload)
        rec = dict(a, explanation=text, llm_model=model)
        out.append(rec)
        if verbose:
            log.info("  [%2d/%d] %s sev=%.2f %s", a["rank"], len(anomalies), a["station_id"],
                     a["severity_score"], "✓" if not text.startswith("Error") else "✗")
    return out


def run_ablation(anomalies: List[dict], retriever, model: str = C.LLM_PRIMARY,
                 n: int = C.N_ABLATION) -> Dict[str, List[dict]]:
    """Four configurations on the same n anomalies (Table 5)."""
    full = make_full_chain(retriever, model)
    llm_only = make_llm_only_chain(model)
    rag_only = make_rag_no_context_chain(retriever, model)
    res = {k: [] for k in C.ABLATION_CONFIGS}
    for a in anomalies[:n]:
        ctx = format_anomaly_context(a)
        q = a["retrieval_query"]
        res["Full Pipeline"].append(dict(a, explanation=_safe(full, {"query": q, "anomaly_context": ctx})))
        res["LLM Only"].append(dict(a, explanation=_safe(llm_only, {"anomaly_context": GENERIC_ANOMALY_CONTEXT})))
        res["LLM + RAG"].append(dict(a, explanation=_safe(rag_only, {"query": q})))
        res["LLM + Context"].append(dict(a, explanation=_safe(llm_only, {"anomaly_context": ctx})))
        log.info("  ablation anomaly %d/%d done", a["rank"], min(n, len(anomalies)))
    return res


def run_llm_comparison(anomalies: List[dict], retriever,
                       models=(C.LLM_PRIMARY, C.LLM_COMPARISON),
                       n: int = C.N_LLM_COMPARISON) -> Dict[str, List[dict]]:
    """Same anomalies, same retrieved chunks, same prompt; only the backbone varies (Table 6)."""
    out = {}
    for m in models:
        chain = make_full_chain(retriever, m)
        log.info("LLM comparison: %s on %d anomalies", m, min(n, len(anomalies)))
        out[m] = generate_explanations(anomalies[:n], chain, model=m)
    return out


def save_explanations(obj, path: Path) -> None:
    save_json(obj, path)
    log.info("saved %s", path)
