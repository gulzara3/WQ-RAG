"""
WQ-RAG central configuration.

Every value here is traceable to the accepted manuscript
(Journal of Environmental Management, JEMA-D-26-16999R1):
  - Methods Section 2 (data, preprocessing, detection, RAG, evaluation)
  - Fig. 2 (system architecture with all hyperparameters)
  - Table 1 (stations) and Table 2 (parameters)

Override the project root with the environment variable WQRAG_ROOT
(defaults to the repository folder).  All paths are derived from it.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("WQRAG_ROOT", REPO_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"              # one CSV per station from NWIS
PROCESSED_DIR = DATA_DIR / "processed"       # cached .npz windows
KB_DIR = PROJECT_ROOT / "knowledge_base"
KB_REGULATIONS_DIR = KB_DIR / "regulations"
KB_TECHNICAL_DIR = KB_DIR / "technical_guides"
KB_CASE_STUDIES_DIR = KB_DIR / "case_studies"
KB_METADATA_DIR = KB_DIR / "station_metadata"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PAPER_RESULTS_DIR = REPO_ROOT / "paper_results"  # published numbers (tables 3-7)


def ensure_dirs() -> None:
    for d in (RAW_DATA_DIR, PROCESSED_DIR, CHROMA_DB_DIR, RESULTS_DIR, MODELS_DIR,
              EXPLANATIONS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study design (Table 1, Section 2.1, Data Availability statement)
# ---------------------------------------------------------------------------
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# Ordered as in Table 1 / Table 3 / Fig. 9 of the paper.
# NOTE: the manuscript (Table 1, Data Availability) lists the White River
# station as USGS 03351000; the earlier code base used 03353200.  The
# manuscript is authoritative, so 03351000 is the default here.  Change it in
# one place if you need the other gauge.
STATIONS = {
    "01646500": {
        "name": "Potomac River, DC",
        "short": "Potomac",
        "label": "Potomac (DC)",
        "region": "East Coast",
        "regime": "Mid-Atlantic urban (combined sewer influence)",
        "sc_range": "200-600",
        "lat": 38.9497, "lon": -77.1275,
        "color": "#1f77b4",
    },
    "03351000": {
        "name": "White River, IN",
        "short": "White",
        "label": "White River (IN)",
        "region": "Midwest",
        "regime": "Midwestern agricultural (glacial till)",
        "sc_range": "400-1,200",
        "lat": 39.9067, "lon": -86.1069,
        "color": "#2ca02c",
    },
    "11447650": {
        "name": "Sacramento River, CA",
        "short": "Sacramento",
        "label": "Sacramento (CA)",
        "region": "West Coast",
        "regime": "Central Valley tidal / agricultural",
        "sc_range": "100-300",
        "lat": 38.4556, "lon": -121.5017,
        "color": "#ff7f0e",
    },
    "14211010": {
        "name": "Clackamas River, OR",
        "short": "Clackamas",
        "label": "Clackamas (OR)",
        "region": "Pacific NW",
        "regime": "Pacific NW volcanic drinking-water source",
        "sc_range": "30-65",
        "lat": 45.3790, "lon": -122.5770,
        "color": "#d62728",
    },
}
STATION_ORDER = list(STATIONS.keys())

# Station used for the ablation study, the multi-LLM comparison, Fig. 4,
# Fig. 5c and Fig. 6 (Sections 2.4, 3.2, 3.3, 3.5, 3.6).
PRIMARY_STATION = "14211010"

# Table 2: USGS parameter codes -> internal column names (this order is the
# feature order of the input tensor x in R^{96 x 5}).
PARAMETERS = {
    "00010": "Temperature_C",
    "00095": "Conductivity_uScm",
    "00300": "DO_mgL",
    "00400": "pH",
    "63680": "Turbidity_FNU",
}
PARAM_ORDER = list(PARAMETERS.values())
PARAM_DISPLAY = {
    "Temperature_C": ("Temperature", "°C"),
    "Conductivity_uScm": ("SC", "µS/cm"),
    "DO_mgL": ("DO", "mg/L"),
    "pH": ("pH", ""),
    "Turbidity_FNU": ("Turbidity", "FNU"),
}

# ---------------------------------------------------------------------------
# Preprocessing (Section 2.2.1, Fig. 2 Stage I)
# ---------------------------------------------------------------------------
USGS_SENTINEL = -999999.0
PHYSICAL_BOUNDS = {              # values outside are set to NaN
    "Temperature_C": (-5.0, 45.0),
    "pH": (2.0, 14.0),
    "DO_mgL": (0.0, 25.0),
    "Turbidity_FNU": (0.0, 5000.0),
    "Conductivity_uScm": (0.0, 100000.0),
}
SAMPLING_MINUTES = 15
MAX_GAP_STEPS = 4                # <= 1 hour forward-filled; longer gaps dropped
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15   # chronological split
WINDOW_SIZE = 96                 # 24 h of 15-min observations
WINDOW_STRIDE = 24               # 6 h  (75 % overlap)

# Ground-truth labelling (Section 2.2.1, paragraph 2) — statistical exceedance
LABEL_SIGMA = 3.0                # window anomalous if any |z| > 3 sigma
EXTREME_EVENT_SIGMA = 4.0        # independent extreme-event catalogue (Table 7)

# ---------------------------------------------------------------------------
# Anomaly detection (Section 2.2.2, Fig. 2 Stage II)
# ---------------------------------------------------------------------------
SEED = 42

PATCHTST = dict(
    patch_len=16,        # 16 steps = 4 h  -> 6 patches per window
    d_model=128,
    n_heads=8,
    n_encoder_layers=3,
    n_decoder_layers=2,
    bottleneck_dim=64,
    dropout=0.2,
)
LSTM_AE = dict(hidden_dim=128, n_layers=2, dropout=0.2)

TRAINING = dict(
    optimizer="AdamW",
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    epochs=100,
    patience=10,                  # early stopping
    scheduler="ReduceLROnPlateau",
    scheduler_patience=5,
    scheduler_factor=0.5,
    grad_clip=1.0,
)

ISOLATION_FOREST = dict(n_estimators=200, contamination=0.05, random_state=SEED, n_jobs=-1)
ONE_CLASS_SVM = dict(kernel="rbf", nu=0.05, gamma="scale")
OCSVM_MAX_TRAIN = 10000          # subsample for tractability

# Adaptive per-station thresholding (Section 2.2.2, last paragraph)
THRESHOLD_ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]   # theta = mu(e) + alpha*sigma(e)
THRESHOLD_PERCENTILES = [90, 92, 95, 97, 99]
THRESHOLD_MIN_PRECISION = 0.65
FIXED_ALPHA = 2.5                # reference fixed threshold (Fig. 6 caption)

# ---------------------------------------------------------------------------
# RAG (Section 2.2.3, Fig. 2 Stage III)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d
CHROMA_COLLECTION = "wq_rag_knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 20
RETRIEVER_LAMBDA = 0.7           # MMR diversity parameter lambda

KB_CATEGORIES = {
    "regulations": ("Regulatory Standards", KB_REGULATIONS_DIR),
    "technical_guides": ("Technical Guides", KB_TECHNICAL_DIR),
    "case_studies": ("Case Studies", KB_CASE_STUDIES_DIR),
    "station_metadata": ("Station Metadata", KB_METADATA_DIR),
}

# ---------------------------------------------------------------------------
# LLM (Section 2.2.4, Fig. 2 Stage IV)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PRIMARY = "llama3:8b"
LLM_COMPARISON = "mistral:7b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

N_EXPLANATIONS_PER_STATION = 50  # 4 x 50 = 200 (Table 4)
N_LLM_COMPARISON = 50            # Clackamas (Table 6, Fig. 8)
N_ABLATION = 5                   # Clackamas (Table 5)

ABLATION_CONFIGS = ["Full Pipeline", "LLM Only", "LLM + RAG", "LLM + Context"]

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
MODEL_ORDER = ["PatchTST", "LSTM-AE", "IF", "OC-SVM"]
MODEL_COLORS = {"PatchTST": "#1f77b4", "LSTM-AE": "#2ca02c", "IF": "#d62728", "OC-SVM": "#ff7f0e"}
