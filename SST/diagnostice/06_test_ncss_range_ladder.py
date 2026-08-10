# =============================================================================
# Script: 06_test_ncss_range_ladder.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-09
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Test increasing OISST time ranges through NCSS without attempting a full year.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import requests
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/"
    "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc"
)

TESTS = [
    ("7_days",  "1982-01-01T00:00:00Z", "1982-01-07T00:00:00Z", 7),
    ("31_days", "1982-01-01T00:00:00Z", "1982-01-31T00:00:00Z", 31),
    ("60_days", "1982-01-01T00:00:00Z", "1982-03-01T00:00:00Z", 60),
    ("90_days", "1982-01-01T00:00:00Z", "1982-03-31T00:00:00Z", 90),
]


def test_range(name: str, start: str, end: str, expected_times: int) -> bool:
    output = OUT_DIR / f"ncss_{name}_med.nc"
    partial = output.with_suffix(".nc.part")
    output.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)

    params = {
        "var": "sst",
        "north": 46.5,
        "south": 30.0,
        "west": -6.5,
        "east": 37.5,
        "time_start": start,
        "time_end": end,
        "accept": "netCDF4",
    }

    print("\n" + "=" * 78)
    print(f"TEST: {name}")
    print(f"{start} -> {end}")

    try:
        with requests.get(URL, params=params, stream=True, timeout=(30, 180)) as r:
            print(f"HTTP status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Content-Length: {r.headers.get('Content-Length')}")
            print(f"Final URL: {r.url}")

            if r.status_code >= 400:
                print("Server response:")
                print(r.text[:4000])
                return False

            downloaded = 0
            with partial.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        partial.replace(output)
        print(f"Downloaded: {downloaded:,} bytes")

        with xr.open_dataset(output) as ds:
            if "sst" not in ds:
                raise RuntimeError(f"'sst' missing; variables={list(ds.data_vars)}")
            n_time = ds.sizes.get("time")
            print(f"Dimensions: {dict(ds.sizes)}")
            print(f"time steps: {n_time}")

            ds["sst"].isel(time=0).load()
            ds["sst"].isel(time=-1).load()

            if n_time != expected_times:
                print(
                    f"WARNING: expected {expected_times} time steps, "
                    f"server returned {n_time}"
                )

        print("PASS")
        return True

    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    for args in TESTS:
        if not test_range(*args):
            print("\nStopping at first NCSS failure.")
            break


if __name__ == "__main__":
    main()
