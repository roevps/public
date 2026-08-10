# =============================================================================
# Script: 1_download_data.py
# Version: 1.7
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-09
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Download/subset four public SST products to the Mediterranean domain.
#
# OUTPUT
# -----------------------------------------------------------------------------
# data/raw/oisst_v21_med.nc, ersst_v5_med.nc, hadsst4_actuals_med.nc, cobe_sst2_med.nc
# =============================================================================

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import requests
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Broad Mediterranean extraction box. A stricter basin mask is applied in script 2.
LAT_MIN, LAT_MAX = 30.0, 46.5
LON_MIN, LON_MAX = -6.5, 37.5

OISST_CHUNK_DAYS = 90

# NOAA's conventional float missing-value sentinel. Keep one missing-data
# convention on disk: _FillValue. In memory, xarray continues to use NaN.
OISST_FILL_VALUE = np.float32(-9.969209968386869e36)

URLS = {
    "ersst": "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc",
    "cobe": "https://psl.noaa.gov/thredds/dodsC/Datasets/COBE2/sst.mon.mean.nc",
    "oisst_dap_template": (
        "https://psl.noaa.gov/thredds/dodsC/"
        "Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    ),
    "oisst_ncss_template": (
        "https://psl.noaa.gov/thredds/ncss/grid/"
        "Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    ),
    "hadsst": (
        "https://hadleyserver.metoffice.gov.uk/hadsst4/data/data/"
        "HadSST.4.2.0.0_actuals_median.nc"
    ),
}


def coord_names(obj: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    lat_name = "lat" if "lat" in obj.coords else "latitude"
    lon_name = "lon" if "lon" in obj.coords else "longitude"
    return lat_name, lon_name


def normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """Return dataset with longitude in [-180, 180) and sorted."""
    _, lon_name = coord_names(ds)
    lon = (((ds[lon_name] + 180.0) % 360.0) - 180.0)
    return ds.assign_coords({lon_name: lon}).sortby(lon_name)


def crop_med(ds: xr.Dataset) -> xr.Dataset:
    """Crop an in-memory/local dataset to the broad Mediterranean box."""
    ds = normalize_lon(ds)
    lat_name, lon_name = coord_names(ds)

    lat = ds[lat_name]
    lat_slice = (
        slice(LAT_MIN, LAT_MAX)
        if float(lat[0]) < float(lat[-1])
        else slice(LAT_MAX, LAT_MIN)
    )

    return ds.sel(
        {
            lat_name: lat_slice,
            lon_name: slice(LON_MIN, LON_MAX),
        }
    )


def prepare_oisst_for_write(ds: xr.Dataset) -> xr.Dataset:
    """Use one consistent missing-data encoding before writing OISST NetCDF."""
    ds = ds.copy(deep=False)
    ds.attrs.pop("_NCProperties", None)

    if "sst" in ds:
        # xarray decodes NetCDF missing values to NaN in memory. After reopening
        # derived files, inherited _FillValue and missing_value can disagree,
        # which prevents a subsequent to_netcdf(). Remove both and write one
        # explicit _FillValue instead.
        ds["sst"].attrs.pop("_FillValue", None)
        ds["sst"].attrs.pop("missing_value", None)
        ds["sst"].encoding.pop("_FillValue", None)
        ds["sst"].encoding.pop("missing_value", None)

        ds["sst"].encoding["dtype"] = "float32"
        ds["sst"].encoding["_FillValue"] = OISST_FILL_VALUE

    return ds


def valid_netcdf(path: Path, variables=("sst",)) -> bool:
    """Return True when an existing NetCDF file opens and contains usable data."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with xr.open_dataset(path) as ds:
            keep = [v for v in variables if v in ds.data_vars]
            if not keep:
                return False

            da = ds[keep[0]]
            if any(size == 0 for size in da.sizes.values()):
                return False

            indexers = {dim: 0 for dim in da.dims}
            da.isel(indexers).load()

        return True

    except Exception:
        return False


def valid_oisst_range(path: Path, start: date, end: date) -> bool:
    """Validate OISST content, date coverage and first/last SST slices."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    expected_count = (end - start).days + 1

    try:
        with xr.open_dataset(path) as ds:
            if "sst" not in ds.data_vars:
                return False

            for dim in ("time",):
                if dim not in ds.sizes or ds.sizes[dim] == 0:
                    return False

            lat_name, lon_name = coord_names(ds)
            if ds.sizes.get(lat_name, 0) == 0 or ds.sizes.get(lon_name, 0) == 0:
                return False

            if ds.sizes["time"] != expected_count:
                return False

            first = np.datetime64(ds["time"].values[0], "D")
            last = np.datetime64(ds["time"].values[-1], "D")
            if first != np.datetime64(start.isoformat()):
                return False
            if last != np.datetime64(end.isoformat()):
                return False

            lon_min = float(ds[lon_name].min())
            lon_max = float(ds[lon_name].max())
            if lon_min < -180.0 or lon_max >= 180.0:
                return False

            # Force real data reads at both ends of the file.
            ds["sst"].isel(time=0).load()
            ds["sst"].isel(time=-1).load()

        return True

    except Exception:
        return False


def remote_subset(url: str, output: Path, variables=("sst",), retries: int = 4) -> None:
    """Read a small Mediterranean subset from OPeNDAP and save it locally."""
    if valid_netcdf(output, variables):
        print(f"Valid file already exists — skipping {output}")
        return

    if output.exists():
        print(f"Existing file is invalid/incomplete — replacing {output}")
        output.unlink()

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Opening {url} — attempt {attempt}/{retries}")
            with xr.open_dataset(url, engine="netcdf4") as ds:
                keep = [v for v in variables if v in ds.data_vars]
                if not keep:
                    raise KeyError(
                        f"None of {variables} found; variables={list(ds.data_vars)}"
                    )
                sub = crop_med(ds[keep]).load()
            break

        except Exception as exc:
            last_error = exc
            print(f"Remote read attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(3 * attempt)

    else:
        raise RuntimeError(f"Failed to read {url}") from last_error

    sub.attrs.pop("_NCProperties", None)

    try:
        sub.to_netcdf(output)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Read {url}, but failed to save local file {output}"
        ) from exc

    if not valid_netcdf(output, variables):
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Created file failed validation: {output}")

    print(f"Saved and validated {output}")


def direct_download(url: str, output: Path, retries: int = 4) -> None:
    """Download a file directly, retrying only errors that may be transient."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} — attempt {attempt}/{retries}")
            with requests.get(url, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                with output.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded {output}")
            return

        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None

            if status is not None and 400 <= status < 500 and status not in (408, 429):
                raise RuntimeError(
                    f"Download URL returned HTTP {status}: {url}"
                ) from exc

            print(f"Attempt {attempt}/{retries} failed: {exc}")

        except requests.RequestException as exc:
            last_error = exc
            print(f"Attempt {attempt}/{retries} failed: {exc}")

        if attempt < retries:
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download {url}") from last_error


def download_hadsst() -> None:
    tmp = DATA_DIR / "hadsst4_actuals_global.nc"
    target = DATA_DIR / "hadsst4_actuals_med.nc"

    if valid_netcdf(target, ("tos",)):
        print(f"Valid file already exists — skipping {target}")
        return

    if target.exists():
        print(f"Existing file is invalid/incomplete — replacing {target}")
        target.unlink()

    if valid_netcdf(tmp, ("tos",)):
        print(f"Using existing valid temporary file {tmp}")
    else:
        if tmp.exists():
            print(f"Existing temporary file is invalid/incomplete — replacing {tmp}")
            tmp.unlink()
        direct_download(URLS["hadsst"], tmp)

    with xr.open_dataset(tmp) as ds:
        sub = crop_med(ds).load()

    sub.attrs.pop("_NCProperties", None)
    sub.to_netcdf(target)

    if not valid_netcdf(target, ("tos",)):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Created HadSST file failed validation: {target}")

    tmp.unlink(missing_ok=True)
    print(f"Saved and validated {target}")


def oisst_available_end(year: int) -> date:
    """Read only the OISST time coordinate to determine the last available day."""
    current_year = datetime.now(timezone.utc).year

    if year < current_year:
        return date(year, 12, 31)

    if year > current_year:
        raise ValueError(
            f"OISST year {year} is in the future; current UTC year is {current_year}."
        )

    url = URLS["oisst_dap_template"].format(year=year)
    print(f"Checking latest available OISST date for {year}")

    try:
        with xr.open_dataset(url, engine="netcdf4") as ds:
            if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
                raise RuntimeError("OISST time coordinate is missing or empty.")

            last = np.datetime64(ds["time"].values[-1], "D")
            return date.fromisoformat(str(last))

    except Exception as exc:
        raise RuntimeError(
            f"Could not determine available OISST dates from {url}"
        ) from exc


def chunk_ranges(start: date, end: date, days: int = OISST_CHUNK_DAYS):
    """Yield inclusive date ranges no longer than `days`."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def oisst_chunk_name(year: int, start: date, end: date) -> str:
    start_doy = start.timetuple().tm_yday
    end_doy = end.timetuple().tm_yday
    return f"oisst_{year}_{start_doy:03d}_{end_doy:03d}_med.nc"


def download_oisst_ncss_chunk(
    year: int,
    start: date,
    end: date,
    output: Path,
    retries: int = 4,
) -> None:
    """Download and locally normalize one OISST NCSS chunk."""
    if valid_oisst_range(output, start, end):
        print(f"Valid OISST chunk already exists — skipping {output.name}")
        return

    if output.exists():
        print(f"Invalid OISST chunk — replacing {output.name}")
        output.unlink()

    raw_part = output.with_suffix(".download")
    processed_part = output.with_suffix(".part")
    raw_part.unlink(missing_ok=True)
    processed_part.unlink(missing_ok=True)

    url = URLS["oisst_ncss_template"].format(year=year)
    params = {
        "var": "sst",
        "north": LAT_MAX,
        "south": LAT_MIN,
        "west": LON_MIN,
        "east": LON_MAX,
        "time_start": f"{start.isoformat()}T00:00:00Z",
        "time_end": f"{end.isoformat()}T00:00:00Z",
        "accept": "netCDF4",
    }

    last_error = None

    for attempt in range(1, retries + 1):
        raw_part.unlink(missing_ok=True)
        processed_part.unlink(missing_ok=True)

        try:
            print(
                f"NCSS OISST {start} -> {end} "
                f"— attempt {attempt}/{retries}"
            )

            with requests.get(
                url,
                params=params,
                stream=True,
                timeout=(30, 300),
            ) as r:
                r.raise_for_status()

                with raw_part.open("wb") as f:
                    for block in r.iter_content(chunk_size=1024 * 1024):
                        if block:
                            f.write(block)

            # Process only after the HTTP transfer is complete.
            with xr.open_dataset(raw_part) as ds:
                if "sst" not in ds.data_vars:
                    raise RuntimeError(
                        f"NCSS response does not contain 'sst': {list(ds.data_vars)}"
                    )

                sub = crop_med(ds[["sst"]]).load()

            sub = prepare_oisst_for_write(sub)
            sub.to_netcdf(processed_part)

            if not valid_oisst_range(processed_part, start, end):
                raise RuntimeError("Downloaded NCSS chunk failed validation.")

            processed_part.replace(output)
            raw_part.unlink(missing_ok=True)
            print(f"Saved and validated {output}")
            return

        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None

            if status is not None and 400 <= status < 500 and status not in (408, 429):
                raw_part.unlink(missing_ok=True)
                processed_part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"NCSS returned HTTP {status} for {start} -> {end}: {r.url}"
                ) from exc

            print(f"NCSS attempt {attempt}/{retries} failed: {exc}")

        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            print(f"NCSS attempt {attempt}/{retries} failed: {exc}")

        raw_part.unlink(missing_ok=True)
        processed_part.unlink(missing_ok=True)

        if attempt < retries:
            time.sleep(3 * attempt)

    raise RuntimeError(
        f"Failed OISST NCSS chunk {start} -> {end}"
    ) from last_error


def combine_oisst_parts(
    parts: list[Path],
    output: Path,
    start: date,
    end: date,
) -> None:
    """Combine validated OISST parts into one validated NetCDF file."""
    if valid_oisst_range(output, start, end):
        print(f"Valid combined OISST file already exists — skipping {output}")
        return

    if output.exists():
        print(f"Invalid combined OISST file — replacing {output}")
        output.unlink()

    tmp = output.with_suffix(".part")
    tmp.unlink(missing_ok=True)

    datasets = [xr.open_dataset(path) for path in parts]

    try:
        combined = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        ).sortby("time").load()

        combined = prepare_oisst_for_write(combined)
        combined.to_netcdf(tmp)

    finally:
        for ds in datasets:
            ds.close()

    if not valid_oisst_range(tmp, start, end):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Combined OISST file failed validation: {output}")

    tmp.replace(output)
    print(f"Saved and validated {output}")


def download_oisst(start_year: int, end_year: int) -> None:
    """Download OISST using resumable 90-day NCSS chunks."""
    if start_year > end_year:
        raise ValueError("--oisst-start cannot be later than --end-year")

    first_date = date(start_year, 1, 1)
    last_date = oisst_available_end(end_year)

    if last_date < first_date:
        raise ValueError("Requested OISST date range is empty.")

    combined_path = DATA_DIR / "oisst_v21_med.nc"

    # If the final product is already complete for the currently available range,
    # do not inspect or download any intermediate parts.
    if valid_oisst_range(combined_path, first_date, last_date):
        print(f"Valid file already exists — skipping {combined_path}")
        return

    temp_dir = DATA_DIR / "_oisst_parts"
    temp_dir.mkdir(exist_ok=True)

    yearly_parts: list[Path] = []

    for year in range(start_year, end_year + 1):
        year_start = date(year, 1, 1)
        year_end = min(date(year, 12, 31), last_date)

        if year_end < year_start:
            break

        yearly_file = temp_dir / f"oisst_{year}_med.nc"

        if valid_oisst_range(yearly_file, year_start, year_end):
            print(f"Valid OISST year already exists — skipping {yearly_file.name}")
            yearly_parts.append(yearly_file)
            continue

        year_dir = temp_dir / str(year)
        year_dir.mkdir(exist_ok=True)

        chunks: list[Path] = []

        for chunk_start, chunk_end in chunk_ranges(year_start, year_end):
            chunk_file = year_dir / oisst_chunk_name(
                year,
                chunk_start,
                chunk_end,
            )

            download_oisst_ncss_chunk(
                year,
                chunk_start,
                chunk_end,
                chunk_file,
            )
            chunks.append(chunk_file)

        combine_oisst_parts(
            chunks,
            yearly_file,
            year_start,
            year_end,
        )
        yearly_parts.append(yearly_file)

    combine_oisst_parts(
        yearly_parts,
        combined_path,
        first_date,
        last_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-year",
        type=int,
        default=1900,
        help="Start year for monthly historical products.",
    )
    parser.add_argument(
        "--oisst-start",
        type=int,
        default=1982,
        help="OISST begins in late 1981; 1982 gives complete years.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now(timezone.utc).year,
    )
    parser.add_argument("--skip-oisst", action="store_true")
    args = parser.parse_args()

    # Monthly products: OPeNDAP transfers only the Mediterranean subset.
    remote_subset(URLS["ersst"], DATA_DIR / "ersst_v5_med.nc")
    remote_subset(URLS["cobe"], DATA_DIR / "cobe_sst2_med.nc")

    # HadSST4 is small enough to download in full, then crop locally.
    download_hadsst()

    if not args.skip_oisst:
        download_oisst(args.oisst_start, args.end_year)

    print("\nDownload complete.")
    print(
        "Note: script 2 applies a stricter Mediterranean basin mask "
        "and time filtering."
    )


if __name__ == "__main__":
    main()