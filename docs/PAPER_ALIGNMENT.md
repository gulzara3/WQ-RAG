# Paper ↔ code alignment

Every quantitative statement in the accepted manuscript is mapped to the line of code that
implements it.  Items marked ⚠ are places where the manuscript is internally inconsistent or
where the previous repository diverged; each has been resolved in favour of the manuscript and
documented so the authors can verify.

## Stage I — data & preprocessing (§2.1-2.2.1, Fig. 2)

| Paper | Code |
|-------|------|
| 4 stations, USGS IDs 01646500 / 03351000 / 14211010 / 11447650 (Table 1, Data Availability) | `config.STATIONS` |
| 5 parameters, codes 00010/00095/00300/00400/63680 (Table 2) | `config.PARAMETERS` |
| NWIS REST-API, 15-min, Jan 2021-Dec 2024 | `data_download.fetch_station` |
| sentinel −999 999 → NaN | `config.USGS_SENTINEL`, `preprocessing.clean_series` |
| bounds T −5..45 °C, pH 2..14, DO 0..25, turbidity 0..5 000 | `config.PHYSICAL_BOUNDS` |
| gaps ≤ 1 h forward-filled, longer excluded | `MAX_GAP_STEPS = 4`, `clean_series` |
| chronological 70/15/15 split, training-set statistics for z-scores | `chronological_split`, `fit_standardiser` (ddof 0) |
| 96-step windows, stride 24 (75 % overlap), x ∈ ℝ⁹⁶ˣ⁵ | `make_windows` |
| labels: any parameter \|z\| > 3σ (train stats) | `label_windows`, `LABEL_SIGMA` |
| extreme events \|z\| > 4σ (Table 7) | `evaluation.extreme_events`, `EXTREME_EVENT_SIGMA` |


## Stage II — detection (§2.2.2, Fig. 2, §3.1-3.3, Tables 3, Fig. 3-6, 9, 10)

| Paper | Code |
|-------|------|
| PatchTST-AE: 16-step patches × 6, d_model 128, 8 heads, 3 enc / 2 dec layers, 64-d bottleneck, dropout 0.2 | `config.PATCHTST`, `models.PatchTSTAutoencoder` |
| LSTM-AE: 2-layer 128-hidden encoder/decoder, dropout 0.2 | `config.LSTM_AE`, `models.LSTMAutoencoder` |
| AdamW, lr 1e-4, wd 1e-5, batch 64, ≤100 epochs, early stop patience 10, ReduceLROnPlateau | `config.TRAINING`, `detection.train_autoencoder` |
| IF 200 trees, contamination 0.05 | `config.ISOLATION_FOREST` |
| OC-SVM RBF, ν 0.05, γ scale | `config.ONE_CLASS_SVM` |
| θ = μ(e)+α·σ(e), α∈{1.0..4.0} and P90-P99, max val-F1 s.t. precision ≥ 0.65 | `thresholding.select_threshold` |
| fixed-α reference (α = 2.5, Fig. 6 caption; +61 % claim) | `thresholding.fixed_threshold`, `F1_fixed_threshold` in detection JSON |
| P, R, F1, CSI (Eq. 1), AUC-ROC, confusion matrices | `detection.detection_metrics` |
| Table 3 | `detection.results_to_table3` → `results/tables/table3_detection_performance.csv` |
| Fig. 3 / 9 / 10 | `figures.fig03_detection_performance`, `fig09_confusion_matrices`, `fig10_best_model` |
| Fig. 4 (Clackamas losses) | `figures.fig04_training_curves` (loss curves stored in `detection_<id>.json`) |
| Fig. 5 (violin, ROC, log-KDE Clackamas, threshold sweep) | `figures.fig05_reconstruction_error`, `thresholding.threshold_sweep` |
| Fig. 6 (Clackamas 5 parameters + error, anomalies shaded) | `figures.fig06_timeseries` — x-axis is the flattened test time-step index (the paper labels it "Window Index") |


## Stage III — RAG (§2.2.3, Fig. 2)

| Paper | Code |
|-------|------|
| 25 documents, 4 categories | `config.KB_CATEGORIES`, `knowledge_base/MANIFEST.md` |
| RecursiveCharacterTextSplitter 1 000 / 200 | `knowledge_base.chunk_documents` |
| all-MiniLM-L6-v2, 384-d, cosine, ChromaDB | `knowledge_base.embedding_model`, `build_vector_store` |
| MMR, k = 6, pool 20, λ = 0.7 | `knowledge_base.make_retriever` |
| query from 3 most-deviated parameters + direction | `explainer.build_retrieval_query` |

## Stage IV — explanation & evaluation (§2.2.4, §2.4, §3.4-3.6, Tables 4-6, Figs 7-8)

| Paper | Code |
|-------|------|
| 5-section prompt (root cause, severity, regulatory, actions, confidence) | `explainer.FULL_PROMPT` |
| Llama-3-8B primary, Mistral-7B comparison, Ollama, temp 0.1, 1 024 tokens, zero-shot | `config.LLM_*`, `explainer._llm` |
| 50 explanations/station = 200 | `N_EXPLANATIONS_PER_STATION`, `scripts/05` |
| severity = (e−θ)/(max e−θ) ∈ [0,1]; peak step; z-scores; physical values | `explainer.extract_anomalies` |
| ablation: Full / LLM Only / LLM + RAG / LLM + Context, Clackamas n = 5 | `explainer.run_ablation`, `config.ABLATION_CONFIGS`, `N_ABLATION` |
| multi-LLM n = 50 Clackamas, same chunks / prompt; paired Cohen's d | `explainer.run_llm_comparison`, `evaluation.cohens_d_paired` |
| completeness = fraction of 5 sections | `evaluation._SECTION_PATTERNS` |
| regulatory accuracy = fraction of 5 checks (EPA, WHO, numeric threshold + units, MCL, named standard) | `evaluation._REGULATORY_CHECKS` |
| actionability = fraction of 4 checks | `evaluation._ACTIONABILITY_CHECKS` |
| Table 4 (mean ± SD per station; cross-station mean ± SD of station means) | `evaluation.table4_explanation_quality` |
| Table 5 | `evaluation.table5_ablation` |
| Table 6 incl. Δ and Cohen's d; wins/ties/losses (34/7/9 in text) | `evaluation.table6_llm_comparison` (`paired.attrs`) |
| Table 7 & parameter breakdown (120/66/34/29/8) | `evaluation.table7_extreme_events`, `parameter_breakdown` |
| Fig. 7 / Fig. 8 | `figures.fig07_explanation_quality`, `fig08_llm_comparison` |


