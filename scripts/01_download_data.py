#!/usr/bin/env python
"""Download 15-min USGS NWIS data for the four stations (Section 2.1)."""
import argparse
import _common  # noqa: F401
from wqrag import config as C
from wqrag.data_download import download_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", nargs="*", default=C.STATION_ORDER, help="USGS site IDs")
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--end", default=C.END_DATE)
    a = ap.parse_args()
    download_all(a.station, a.start, a.end)
