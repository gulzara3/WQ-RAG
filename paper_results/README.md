# Published results (for verification and figure regeneration)

These CSVs transcribe Tables 1, 3, 4, 5, 6 and 7 exactly as they appear in the
accepted manuscript.  They let you

* regenerate Figs 3, 7, 8(a), 9 and 10 without re-running the experiments
  (`python scripts/09_make_figures_tables.py --from-paper`), and
* compare a fresh run (`results/tables/*.csv`) against the published numbers
  (`python scripts/10_compare_with_paper.py`).

Figs 4, 5, 6 and 8(b) need raw per-window scores / per-anomaly pairs, which are
produced only by an actual run.

The White River station is listed as USGS 03351000 (Table 1 and the Data
Availability statement of the manuscript).
