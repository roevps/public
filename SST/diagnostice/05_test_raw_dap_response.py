# =============================================================================
# Script: 05_test_raw_dap_response.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-09
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Inspect NOAA's raw HTTP response for OPeNDAP requests that pass and fail in xarray.
# =============================================================================

from __future__ import annotations

import requests

BASE = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc.dods"
)

TESTS = {
    "7_days_east_med": "sst[0:1:6][480:1:545][0:1:149]",
    "30_days_east_med": "sst[0:1:29][480:1:545][0:1:149]",
    "90_days_east_med": "sst[0:1:89][480:1:545][0:1:149]",
    "365_days_east_med": "sst[0:1:364][480:1:545][0:1:149]",
}


def run(name: str, constraint: str) -> None:
    url = f"{BASE}?{constraint}"
    print("\n" + "=" * 78)
    print(name)
    print(url)

    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as r:
            print(f"HTTP status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Content-Length: {r.headers.get('Content-Length')}")
            print(f"XDODS-Server: {r.headers.get('XDODS-Server')}")
            print(f"Server: {r.headers.get('Server')}")

            first = next(r.iter_content(chunk_size=4096), b"")
            print(f"First response bytes: {len(first):,}")

            if r.status_code >= 400:
                text = first.decode("utf-8", errors="replace")
                print("Server response:")
                print(text[:4000])
            else:
                print("PASS: server started returning a DAP binary response.")
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")


def main() -> None:
    for name, constraint in TESTS.items():
        run(name, constraint)


if __name__ == "__main__":
    main()
