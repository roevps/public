# OISST access diagnostics

Run from `C:\Repos\SST` in this order:

```powershell
python 00_check_http_services.py
python 01_test_opendap_metadata.py
python 02_test_opendap_read_ladder.py
python 03_test_ncss_one_day.py
python 04_test_ncss_full_year.py
```

## What each test answers

1. `00_check_http_services.py`
   - Does the named 1982 file exist?
   - Are the catalog, OPeNDAP metadata, NCSS description, and direct HTTP services reachable?
   - It does **not** download the 482 MB source file.

2. `01_test_opendap_metadata.py`
   - Can `netCDF4/xarray` open the OPeNDAP dataset and read its metadata?
   - It does not intentionally load the SST cube.

3. `02_test_opendap_read_ladder.py`
   - At what request size does actual OPeNDAP SST reading fail?
   - Tests a scalar, 10x10 block, one-day Mediterranean pieces, 7 days, then 365 days.
   - No NetCDF output file is created.

4. `03_test_ncss_one_day.py`
   - Can NOAA's NetCDF Subset Service return one Mediterranean day as a real NetCDF file?
   - Output: `data/diagnostics/ncss_1982_01_01_med.nc`

5. `04_test_ncss_full_year.py`
   - Can NCSS return the complete 1982 Mediterranean year?
   - Run only after script 03 succeeds.
   - Output: `data/diagnostics/ncss_1982_full_year_med.nc`
   - A failed/incomplete transfer remains as `.nc.part` for inspection.

## Decision logic

- 00 fails: endpoint/service/network problem.
- 00 passes, 01 fails: local netCDF4/OPeNDAP client compatibility problem.
- 01 passes, 02 scalar fails: OPeNDAP data service/client problem even for minimal reads.
- Small 02 tests pass but larger tests fail: OPeNDAP request-size/server processing problem.
- 03 passes: NCSS is a viable replacement for OPeNDAP subsetting.
- 03 passes and 04 passes: use NCSS for the production OISST yearly downloader.
- 03 fails too: inspect HTTP status/body before changing production code.
