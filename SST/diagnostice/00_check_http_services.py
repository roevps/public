# =============================================================================
# Script: 00_check_http_services.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Check that the 1982 OISST file and NOAA THREDDS access services are reachable without downloading the dataset.
# =============================================================================

from __future__ import annotations

import requests

DATASET = "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc"
BASE = "https://psl.noaa.gov/thredds"

URLS = {
    "catalog": (
        "https://psl.noaa.gov/thredds/catalog/"
        "Datasets/noaa.oisst.v2.highres/catalog.html"
        "?dataset=Datasets%2Fnoaa.oisst.v2.highres%2Fsst.day.mean.1982.nc"
    ),
    "opendap_dds": f"{BASE}/dodsC/{DATASET}.dds",
    "opendap_das": f"{BASE}/dodsC/{DATASET}.das",
    "ncss_description": f"{BASE}/ncss/grid/{DATASET}/dataset.xml",
    "http_file": f"{BASE}/fileServer/{DATASET}",
}


def check_get(name: str, url: str) -> None:
    try:
        r = requests.get(url, timeout=30)
        print(f"{name:18s} status={r.status_code} bytes={len(r.content):,}")
        r.raise_for_status()
    except Exception as exc:
        print(f"{name:18s} FAIL: {type(exc).__name__}: {exc}")


def check_http_file(url: str) -> None:
    """Request only the first byte so the 482 MB file is not downloaded."""
    try:
        headers = {"Range": "bytes=0-0"}
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            print(
                f"{'http_file':18s} status={r.status_code} "
                f"content-range={r.headers.get('Content-Range')} "
                f"content-length={r.headers.get('Content-Length')}"
            )
            if r.status_code not in (200, 206):
                r.raise_for_status()

            # Consume at most one small chunk, then close the connection.
            next(r.iter_content(chunk_size=16), b"")
    except Exception as exc:
        print(f"{'http_file':18s} FAIL: {type(exc).__name__}: {exc}")


def main() -> None:
    print("OISST 1982 service availability")
    print("=" * 72)

    for name in ("catalog", "opendap_dds", "opendap_das", "ncss_description"):
        check_get(name, URLS[name])

    check_http_file(URLS["http_file"])


if __name__ == "__main__":
    main()
