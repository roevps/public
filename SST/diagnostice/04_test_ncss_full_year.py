# =============================================================================
# Script: 04_test_ncss_full_year.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Request the full 1982 Mediterranean OISST subset through NOAA NCSS and validate the completed NetCDF file.
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

OUTPUT = OUT_DIR / "ncss_1982_full_year_med.nc"
PARTIAL = OUTPUT.with_suffix(".nc.part")

PARAMS = {
    "var": "sst",
    "north": 46.5,
    "south": 30.0,
    "west": -6.5,
    "east": 37.5,
    "time": "all",
    "accept": "netCDF4",
}


def main() -> None:
    print("NCSS full-year Mediterranean test")
    print(f"Endpoint: {URL}")
    print(f"Parameters: {PARAMS}")

    PARTIAL.unlink(missing_ok=True)

    try:
        with requests.get(URL, params=PARAMS, stream=True, timeout=(30, 300)) as r:
            print(f"HTTP status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Content-Length: {r.headers.get('Content-Length')}")
            print(f"Final URL: {r.url}")
            r.raise_for_status()

            downloaded = 0
            with PARTIAL.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rReceived {downloaded / (1024 * 1024):.1f} MiB", end="")

        print()
        PARTIAL.replace(OUTPUT)

    except Exception:
        print()
        print(f"FAIL: partial file retained for inspection: {PARTIAL}")
        raise

    print(f"Saved: {OUTPUT}")
    print(f"Downloaded bytes: {OUTPUT.stat().st_size:,}")

    with xr.open_dataset(OUTPUT) as ds:
        print("PASS: returned file opens as NetCDF")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Variables: {list(ds.data_vars)}")

        if "sst" not in ds:
            raise RuntimeError("'sst' missing from NCSS response")

        if ds.sizes.get("time") != 365:
            raise RuntimeError(
                f"Expected 365 time steps for 1982, got {ds.sizes.get('time')}"
            )

        first = ds["sst"].isel(time=0).load()
        last = ds["sst"].isel(time=-1).load()
        print(f"First time: {ds['time'].values[0]}")
        print(f"Last time:  {ds['time'].values[-1]}")
        print(f"First-day grid shape: {first.shape}")
        print(f"Last-day grid shape:  {last.shape}")

    print("\nPASS: NCSS can deliver the full 1982 Mediterranean subset.")


if __name__ == "__main__":
    main()
