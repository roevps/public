# =============================================================================
# Script: 07_test_opendap_time_threshold.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-09
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Find the approximate OPeNDAP time-span threshold for an east-Mediterranean SST subset.
# =============================================================================

from __future__ import annotations

import numpy as np
import xarray as xr

URL = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc"
)

DAY_COUNTS = [7, 14, 21, 31, 45, 60, 90, 120, 180, 270, 365]


def main() -> None:
    with xr.open_dataset(URL, engine="netcdf4") as ds:
        east = ds["sst"].sel(
            lat=slice(30.0, 46.5),
            lon=slice(0.0, 37.5),
        )

        for days in DAY_COUNTS:
            print(f"\nTEST: {days} days")
            try:
                arr = east.isel(time=slice(0, days)).load().values
                arr = np.asarray(arr)
                print(
                    f"PASS: shape={arr.shape}, "
                    f"payload={arr.nbytes / (1024 * 1024):.2f} MiB"
                )
            except Exception as exc:
                print(f"FAIL: {type(exc).__name__}: {exc}")
                print(f"\nFirst failure occurred at {days} days.")
                break


if __name__ == "__main__":
    main()
