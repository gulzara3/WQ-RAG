#!/usr/bin/env python
"""Chunk, embed and index the 25-document knowledge base into ChromaDB (Section 2.2.3)."""
import argparse
import _common  # noqa: F401
from wqrag.knowledge_base import build_rag_backend, kb_inventory
from wqrag.utils import get_logger

log = get_logger("kb")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the vector store")
    ap.add_argument("--test-query", default="water quality anomaly: Turbidity_FNU above normal, DO_mgL below normal")
    a = ap.parse_args()
    inv = kb_inventory()
    for cat, files in inv.items():
        log.info("%-20s %2d files", cat, len(files))
    vs, retriever = build_rag_backend(rebuild=a.rebuild)
    docs = retriever.invoke(a.test_query)
    log.info("Test query -> %d chunks", len(docs))
    for i, d in enumerate(docs, 1):
        log.info("  [%d] %s | %s | %s...", i, d.metadata.get("category"), d.metadata.get("source"),
                 d.page_content[:90].replace("\n", " "))
