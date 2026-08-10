# =============================================================================
# Script: 02_test_opendap_read_ladder.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Increase OPeNDAP SST read size step by step to identify the smallest request that triggers the DAP failure.
# =============================================================================

from __future__ import annotations

import numpy as np
import xarray as xr

URL = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc"
)


def run_test(name: str, operation) -> bool:
    print(f"\nTEST: {name}")
    try:
        result = operation()
        arr = np.asarray(result)
        print(
            f"PASS: shape={arr.shape}, dtype={arr.dtype}, "
            f"bytes={arr.nbytes:,}"
        )
        return True
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    with xr.open_dataset(URL, engine="netcdf4") as ds:
        sst = ds["sst"]

        # 1. One single grid cell on one day.
        run_test(
            "1 scalar: one day, one grid cell",
            lambda: sst.isel(time=0, lat=500, lon=50).load().values,
        )

        # 2. Small contiguous block on one day.
        run_test(
            "10 x 10 cells on one day",
            lambda: sst.isel(
                time=0,
                lat=slice(480, 490),
                lon=slice(0, 10),
            ).load().values,
        )

        # 3. Eastern Mediterranean, one day: contiguous 0..37.5 E request.
        east = sst.sel(lat=slice(30.0, 46.5), lon=slice(0.0, 37.5))
        run_test(
            "Eastern Mediterranean on one day",
            lambda: east.isel(time=0).load().values,
        )

        # 4. Western Mediterranean, one day: 353.5..360 E.
        west = sst.sel(lat=slice(30.0, 46.5), lon=slice(353.5, 360.0))
        run_test(
            "Western Mediterranean on one day",
            lambda: west.isel(time=0).load().values,
        )

        # 5. Eastern Mediterranean for 7 days.
        run_test(
            "Eastern Mediterranean for 7 days",
            lambda: east.isel(time=slice(0, 7)).load().values,
        )

        # 6. Eastern Mediterranean for all 365 days.
        run_test(
            "Eastern Mediterranean for all 365 days",
            lambda: east.load().values,
        )

        # 7. Western Mediterranean for all 365 days.
        run_test(
            "Western Mediterranean for all 365 days",
            lambda: west.load().values,
        )

    print(
        "\nInterpretation: the first FAIL identifies the request scale/path "
        "at which OPeNDAP stops working."
    )


if __name__ == "__main__":
    main()
