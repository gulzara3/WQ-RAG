#!/usr/bin/env python
"""
WQ-RAG — one-shot pipeline runner.

Runs the numbered scripts in order.  LLM stages (05-07) are skipped automatically
if Ollama is not reachable, so the detection half can be reproduced on any machine.

    python run_pipeline.py                 # everything
    python run_pipeline.py --skip-download # data already in data/raw/
    python run_pipeline.py --detection-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "scripts"

STAGES = [
    ("download", "01_download_data.py"),
    ("preprocess", "02_preprocess.py"),
    ("train", "03_train_detectors.py"),
    ("kb", "04_build_knowledge_base.py"),
    ("explain", "05_generate_explanations.py"),
    ("ablation", "06_run_ablation.py"),
    ("compare", "07_run_llm_comparison.py"),
    ("evaluate", "08_evaluate.py"),
    ("figures", "09_make_figures_tables.py"),
]


def run(script: str) -> int:
    print(f"\n{'=' * 72}\n  {script}\n{'=' * 72}", flush=True)
    return subprocess.call([sys.executable, str(S / script)], cwd=HERE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--detection-only", action="store_true", help="stop after training + figures")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from wqrag.explainer import check_ollama
    llm_ok = False if a.detection_only else check_ollama()

    for name, script in STAGES:
        if name == "download" and a.skip_download:
            continue
        if name in ("kb", "explain", "ablation", "compare") and not llm_ok:
            print(f"[skip] {script} (Ollama/LLM not available)")
            continue
        if run(script) != 0:
            print(f"\nStage '{name}' failed — fix and re-run from this stage.")
            sys.exit(1)
    print("\nPipeline complete. See results/tables and results/figures.")
