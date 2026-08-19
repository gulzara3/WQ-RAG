#!/usr/bin/env python
"""Clean, split, standardise, window and label all stations (Section 2.2.1)."""
import argparse
import _common  # noqa: F401
from wqrag import config as C
from wqrag.preprocessing import prepare_all, station_table
from wqrag.utils import get_logger, save_json

log = get_logger("preprocess")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached .npz windows")
    a = ap.parse_args()
    C.ensure_dirs()
    data = prepare_all(use_cache=not a.no_cache)
    t1 = station_table(data)
    t1.to_csv(C.TABLES_DIR / "table1_stations.csv", index=False)
    save_json({k: v.summary() for k, v in data.items()}, C.RESULTS_DIR / "preprocessing_summary.json")
    log.info("\n%s", t1.to_string(index=False))
