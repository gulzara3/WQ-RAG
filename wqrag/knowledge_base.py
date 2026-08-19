"""
Stage III — Retrieval-augmented generation back-end (Section 2.2.3; Fig. 2).

  * Knowledge base: 25 documents in four categories
        regulations / technical_guides / case_studies / station_metadata
    (see knowledge_base/MANIFEST.md for the list and where to obtain each file).
  * Chunking: RecursiveCharacterTextSplitter, 1,000 characters, 200 overlap.
  * Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-d, cosine).
  * Vector store: ChromaDB (persistent).
  * Retrieval: Maximal Marginal Relevance, k = 6, candidate pool 20, lambda = 0.7.

LangChain / Chroma are imported lazily so the detection half of the pipeline
can run without them installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from . import config as C
from .utils import get_logger

log = get_logger(__name__)

SUPPORTED = (".pdf", ".txt", ".md", ".html")


# ---------------------------------------------------------------------------
def load_documents(kb_dirs: dict = C.KB_CATEGORIES) -> List:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs = []
    n_files = 0
    for key, (category, folder) in kb_dirs.items():
        folder = Path(folder)
        if not folder.exists():
            log.warning("KB folder missing: %s", folder)
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in SUPPORTED or f.name.startswith(("_", ".")) \
                    or f.name.upper().startswith("README") or f.name.upper().startswith("MANIFEST"):
                continue
            try:
                loader = PyPDFLoader(str(f)) if f.suffix.lower() == ".pdf" \
                    else TextLoader(str(f), encoding="utf-8", autodetect_encoding=True)
                pages = loader.load()
            except Exception as exc:  # noqa: BLE001
                log.warning("  failed to load %s: %s", f.name, exc)
                continue
            for p in pages:
                p.metadata.update(source=f.name, category=category, category_key=key)
            docs.extend(pages)
            n_files += 1
        log.info("  %-20s %s", category, folder)
    log.info("Loaded %d files -> %d pages", n_files, len(docs))
    return docs


def chunk_documents(docs: List, size: int = C.CHUNK_SIZE, overlap: int = C.CHUNK_OVERLAP) -> List:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""],
                                              length_function=len)
    chunks = splitter.split_documents(docs)
    log.info("Chunked into %d chunks (size %d, overlap %d)", len(chunks), size, overlap)
    return chunks


def embedding_model(device: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings
    from .utils import get_device
    return HuggingFaceEmbeddings(model_name=C.EMBEDDING_MODEL,
                                 model_kwargs={"device": device or get_device()},
                                 encode_kwargs={"normalize_embeddings": True, "batch_size": 32})


def build_vector_store(chunks: List | None = None, persist_dir: Path = C.CHROMA_DB_DIR,
                       rebuild: bool = False):
    """Create (or load) the persistent ChromaDB collection."""
    from langchain_chroma import Chroma
    emb = embedding_model()
    persist_dir = Path(persist_dir)
    exists = (persist_dir / "chroma.sqlite3").exists()

    if exists and not rebuild:
        vs = Chroma(persist_directory=str(persist_dir), collection_name=C.CHROMA_COLLECTION,
                    embedding_function=emb, collection_metadata={"hnsw:space": "cosine"})
        n = vs._collection.count()
        if n > 0:
            log.info("Loaded existing vector store with %d chunks", n)
            return vs
        log.info("Existing store is empty — rebuilding")

    if chunks is None:
        chunks = chunk_documents(load_documents())
    if not chunks:
        raise RuntimeError("No documents found in knowledge_base/. See knowledge_base/MANIFEST.md")

    vs = None
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        if vs is None:
            vs = Chroma.from_documents(batch, emb, persist_directory=str(persist_dir),
                                       collection_name=C.CHROMA_COLLECTION,
                                       collection_metadata={"hnsw:space": "cosine"})
        else:
            vs.add_documents(batch)
    log.info("Vector store built: %d chunks at %s", vs._collection.count(), persist_dir)
    return vs


def make_retriever(vectorstore, k: int = C.RETRIEVER_K, fetch_k: int = C.RETRIEVER_FETCH_K,
                   lambda_mult: float = C.RETRIEVER_LAMBDA):
    return vectorstore.as_retriever(search_type="mmr",
                                    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})


def build_rag_backend(rebuild: bool = False):
    """Convenience: returns (vectorstore, retriever)."""
    vs = build_vector_store(rebuild=rebuild)
    return vs, make_retriever(vs)


def format_docs(docs: List) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[Document {i} | Source: {d.metadata.get('source', '?')} | "
                     f"Category: {d.metadata.get('category', '?')}]\n{d.page_content}")
    return "\n\n".join(parts)


def kb_inventory() -> dict:
    """Count files per category (for README / sanity check of the 25-doc KB)."""
    inv = {}
    for key, (category, folder) in C.KB_CATEGORIES.items():
        folder = Path(folder)
        files = [f.name for f in sorted(folder.iterdir())
                 if folder.exists() and f.suffix.lower() in SUPPORTED
                 and not f.name.startswith(("_", ".")) and not f.name.upper().startswith(("README", "MANIFEST"))] \
            if folder.exists() else []
        inv[category] = files
    return inv
