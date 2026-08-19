# data/

* `raw/station_<USGS_ID>.csv` — created by `scripts/01_download_data.py` from the USGS NWIS
  Instantaneous-Values REST API (15-min, 2021-01-01 … 2024-12-31). Columns:
  Temperature_C, Conductivity_uScm, DO_mgL, pH, Turbidity_FNU (+ `_cd` qualifiers).
* `processed/station_<USGS_ID>.npz` — cached standardised windows, labels and training
  statistics created by `scripts/02_preprocess.py`.

Both folders are git-ignored; the data are public (USGS NWIS, DOI 10.5066/F7P55KJN).
