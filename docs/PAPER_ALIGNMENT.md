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

⚠ **White River station ID.** Table 1 and the Data Availability statement give **03351000**; the
published Fig. 1 map is labelled **03353200**, and the earlier code used 03353200 ("White River at
Nora").  USGS 03351000 *is* "White River near Nora, IN", so the tables are consistent with the
station description and the map label appears to be a typo.  Default here = 03351000; change one
line in `config.STATIONS` if needed.

⚠ **Labels.** The earlier code injected synthetic anomalies into the test set; the manuscript
describes a statistical-exceedance criterion (±3σ on training-period mean).  The code now follows
the manuscript.  The earlier code also fitted the scaler on the whole record and windowed before
splitting; both are fixed.

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

⚠ The earlier code used `Adam`, IF with 300 trees / 0.03, OC-SVM ν 0.03, and an unused
`ADAPTIVE_ALPHAS` list; all now follow Fig. 2.

⚠ Fig. 5b in the published figure reports AUC 0.843-0.903 while Table 3 reports 0.860-0.906 (the
text cites both).  The code computes AUC once (`detection_metrics`) and uses it in both places; the
published discrepancy is a manuscript matter, not a code one.

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

⚠ Table 4 gives the cross-station SDs as 0.068 / 0.012 / 0.020 (sample SD of the four station
means, ddof = 1); Fig. 7d and §3.4 text quote 0.059 / 0.010 / 0.017 (population SD, ddof = 0).
The code uses ddof = 1 and therefore reproduces **Table 4**.  Change `std(ddof=1)` → `std(ddof=0)`
in `table4_explanation_quality` to reproduce the figure's variant.

⚠ The earlier `run_pipeline.py` ran the ablation on whichever station had the most training
windows; the manuscript specifies Clackamas. `config.PRIMARY_STATION = "14211010"` now governs the
ablation, the multi-LLM comparison and Figs 4, 5c, 6.

## Figures not produced by code

Fig. 2 (system architecture) is a drawn diagram: `docs/figures/fig02_system_architecture.png`.
The published Fig. 1 is kept as `docs/figures/fig01_station_map_published.png`; the code version
(`figures.fig01_station_map`) draws the same map (cartopy basemap if installed).

## Verification performed

* `pytest -q` — 14 tests pass (preprocessing semantics, threshold selection, metrics/CSI, baselines
  with paper hyper-parameters, PatchTST/LSTM-AE construction and a 2-epoch CPU training, evaluation
  regexes, Cohen's d, extreme-event table, retrieval-query construction).
* End-to-end smoke run of stages 2, 3, 8, 9 on synthetic 15-min data for all four stations (CPU,
  3 epochs) — all 16 detectors train, Tables 1/3/7 and Figs 1, 3-6, 9, 10 are written.
* `09_make_figures_tables.py --from-paper` regenerates Figs 3, 7, 9, 10 (and 8 with synthesised
  paired differences) from `paper_results/`; Figs 3, 7 and 9 were checked value-by-value against
  the manuscript figures.
