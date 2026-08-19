#!/usr/bin/env python
"""Train/evaluate PatchTST-AE, LSTM-AE, IF and OC-SVM per station (Section 2.2.2; Table 3)."""
import argparse
import _common  # noqa: F401
from wqrag import config as C
from wqrag.detection import results_to_table3, run_station
from wqrag.preprocessing import prepare_all
from wqrag.utils import get_logger, load_json

log = get_logger("train")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", nargs="*", default=C.STATION_ORDER)
    ap.add_argument("--models", nargs="*", default=["PatchTST", "LSTM-AE", "IF", "OC-SVM"])
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke tests)")
    a = ap.parse_args()
    C.ensure_dirs()
    cfg = dict(C.TRAINING)
    if a.epochs:
        cfg["epochs"] = a.epochs
    data = prepare_all(a.station)
    all_res = {}
    for sid, sd in data.items():
        all_res[sid] = run_station(sd, tuple(a.models), train_cfg=cfg)
    # merge with any previously-trained stations
    for sid in C.STATION_ORDER:
        p = C.RESULTS_DIR / f"detection_{sid}.json"
        if sid not in all_res and p.exists():
            all_res[sid] = load_json(p)
    t3 = results_to_table3(all_res)
    t3.to_csv(C.TABLES_DIR / "table3_detection_performance.csv", index=False)
    log.info("\n%s", t3.round(3).to_string(index=False))
