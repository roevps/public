# =============================================================================
# Script: 03_test_ncss_one_day.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Request one Mediterranean OISST day through NOAA NCSS and validate the returned NetCDF file.
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

OUTPUT = OUT_DIR / "ncss_1982_01_01_med.nc"

PARAMS = {
    "var": "sst",
    "north": 46.5,
    "south": 30.0,
    "west": -6.5,
    "east": 37.5,
    "time": "1982-01-01T00:00:00Z",
    "accept": "netCDF4",
}


def main() -> None:
    print("NCSS one-day Mediterranean test")
    print(f"Endpoint: {URL}")
    print(f"Parameters: {PARAMS}")

    try:
        with requests.get(URL, params=PARAMS, stream=True, timeout=120) as r:
            print(f"HTTP status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Content-Length: {r.headers.get('Content-Length')}")
            print(f"Final URL: {r.url}")
            r.raise_for_status()

            with OUTPUT.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except Exception:
        OUTPUT.unlink(missing_ok=True)
        raise

    print(f"Saved: {OUTPUT}")
    print(f"Downloaded bytes: {OUTPUT.stat().st_size:,}")

    try:
        with xr.open_dataset(OUTPUT) as ds:
            print("PASS: returned file opens as NetCDF")
            print(f"Dimensions: {dict(ds.sizes)}")
            print(f"Variables: {list(ds.data_vars)}")

            if "sst" not in ds:
                raise RuntimeError("'sst' missing from NCSS response")

            sample = ds["sst"].isel(time=0).load()
            print(
                f"SST slice loaded: shape={sample.shape}, "
                f"min={float(sample.min(skipna=True)):.3f}, "
                f"max={float(sample.max(skipna=True)):.3f}"
            )

    except Exception:
        print("FAIL: HTTP returned data, but it is not the expected usable NetCDF.")
        raise


if __name__ == "__main__":
    main()
