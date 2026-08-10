# =============================================================================
# Script: 01_test_opendap_metadata.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Open the OISST 1982 OPeNDAP dataset and inspect metadata without requesting SST array values.
# =============================================================================

from __future__ import annotations

import xarray as xr

URL = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.oisst.v2.highres/sst.day.mean.1982.nc"
)


def main() -> None:
    print(f"Opening metadata:\n{URL}\n")

    with xr.open_dataset(URL, engine="netcdf4") as ds:
        print("PASS: xr.open_dataset()")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {list(ds.data_vars)}")
        print(f"Coordinates: {list(ds.coords)}")

        if "sst" not in ds.data_vars:
            raise RuntimeError(f"'sst' not found. Variables: {list(ds.data_vars)}")

        sst = ds["sst"]
        print(f"SST dimensions: {sst.dims}")
        print(f"SST shape: {sst.shape}")
        print(f"SST dtype: {sst.dtype}")

        print("\nCoordinate endpoints (metadata only):")
        print(f"time: {ds['time'].values[0]} -> {ds['time'].values[-1]}")
        print(f"lat:  {float(ds['lat'][0]):.3f} -> {float(ds['lat'][-1]):.3f}")
        print(f"lon:  {float(ds['lon'][0]):.3f} -> {float(ds['lon'][-1]):.3f}")


if __name__ == "__main__":
    main()
