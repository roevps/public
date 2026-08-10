# =============================================================================
# Script: 2_calculate_trends.py
# Version: 1.1
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Calculate area-weighted Mediterranean SST climatologies, anomalies and trends.
#
# INPUT
# -----------------------------------------------------------------------------
# data/raw/*.nc — created by 1_download_data.py
#
# OUTPUT
# -----------------------------------------------------------------------------
# data/processed/monthly_basin_sst.csv, annual_basin_sst.csv, trend_summary.csv,
# data/processed/oisst_daily_basin.csv, oisst_grid_trend.nc
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
RAW = SCRIPT_DIR / "data" / "raw"
OUT = SCRIPT_DIR / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_START = "1982-01-01"
BASELINE_END = "2011-12-31"
TREND_START = "1982-01-01"

FILES = {
    "OISST v2.1": RAW / "oisst_v21_med.nc",
    "ERSST v5": RAW / "ersst_v5_med.nc",
    "HadSST4": RAW / "hadsst4_actuals_med.nc",
    "COBE-SST2": RAW / "cobe_sst2_med.nc",
}


def coord_names(obj):
    lat = "lat" if "lat" in obj.coords else "latitude"
    lon = "lon" if "lon" in obj.coords else "longitude"
    return lat, lon


def find_sst(ds: xr.Dataset) -> xr.DataArray:
    preferred = ["sst", "tos", "sst_median", "temperature_anomaly"]
    for name in preferred:
        if name in ds.data_vars:
            return ds[name]
    candidates = [v for v in ds.data_vars if "sst" in v.lower() or "temperature" in v.lower()]
    if not candidates:
        raise KeyError(f"Could not identify SST variable. Available: {list(ds.data_vars)}")
    return ds[candidates[0]]


def strict_med_mask(da: xr.DataArray) -> xr.DataArray:
    """
    Practical Mediterranean mask for gridded SST.

    SST products already mask land. These rules mainly remove Atlantic cells
    west of Gibraltar and Black Sea cells that fall inside the extraction box.
    """
    lat_name, lon_name = coord_names(da)
    lat = da[lat_name]
    lon = da[lon_name]
    LON, LAT = xr.broadcast(lon, lat)
    LON = LON.transpose(lat_name, lon_name)
    LAT = LAT.transpose(lat_name, lon_name)

    mask = (
        (LAT >= 30.0) & (LAT <= 46.0) &
        (LON >= -6.0) & (LON <= 36.5)
    )

    # Atlantic west of Strait of Gibraltar / Gulf of Cadiz.
    mask &= ~((LON < -5.0) & (LAT > 35.9))
    mask &= ~((LON < -1.5) & (LAT > 43.2))

    # Black Sea and Sea of Azov; retain Aegean.
    mask &= ~((LON > 26.7) & (LAT > 40.9))
    mask &= ~((LON > 29.0) & (LAT > 40.4))

    return mask


def basin_mean(da: xr.DataArray) -> xr.DataArray:
    lat_name, lon_name = coord_names(da)
    mask = strict_med_mask(da)
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.where(mask).weighted(weights).mean((lat_name, lon_name), skipna=True)


def monthly_from_dataset(path: Path, name: str) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        da = find_sst(ds).squeeze(drop=True)
        ts = basin_mean(da)
        # Convert daily products to monthly; monthly inputs are unchanged.
        monthly = ts.resample(time="MS").mean(skipna=True).load()

    df = monthly.to_dataframe(name="sst_c").reset_index()
    df["dataset"] = name
    df["date"] = pd.to_datetime(df["time"])
    return df[["dataset", "date", "sst_c"]].dropna()


def ols_trend(df: pd.DataFrame, start="1982-01-01") -> dict:
    xdf = df[df["date"] >= start].dropna(subset=["sst_anomaly_c"]).copy()
    xdf["year_decimal"] = (
        xdf["date"].dt.year +
        (xdf["date"].dt.dayofyear - 1) /
        np.where(xdf["date"].dt.is_leap_year, 366, 365)
    )
    X = sm.add_constant(xdf["year_decimal"].to_numpy())
    y = xdf["sst_anomaly_c"].to_numpy()

    # HAC errors are more appropriate than iid OLS errors for monthly climate data.
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    slope_decade = model.params[1] * 10.0
    ci = model.conf_int(alpha=0.05)[1] * 10.0

    return {
        "n_months": len(xdf),
        "trend_c_per_decade": slope_decade,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
    }


def add_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dataset, g in monthly.groupby("dataset", sort=False):
        g = g.sort_values("date").copy()
        base = g[(g["date"] >= BASELINE_START) & (g["date"] <= BASELINE_END)]
        clim = base.groupby(base["date"].dt.month)["sst_c"].mean()
        g["month"] = g["date"].dt.month
        g["climatology_1982_2011_c"] = g["month"].map(clim)
        g["sst_anomaly_c"] = g["sst_c"] - g["climatology_1982_2011_c"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def calculate_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    x = monthly.copy()
    x["year"] = x["date"].dt.year
    annual = (
        x.groupby(["dataset", "year"], as_index=False)
         .agg(
             sst_c=("sst_c", "mean"),
             sst_anomaly_c=("sst_anomaly_c", "mean"),
             months=("sst_c", "count"),
         )
    )
    # Keep complete years except current/incomplete year, which remains visible but flagged.
    annual["complete_year"] = annual["months"] == 12
    return annual


def calculate_oisst_daily() -> None:
    path = FILES["OISST v2.1"]
    if not path.exists():
        return
    with xr.open_dataset(path) as ds:
        da = find_sst(ds).squeeze(drop=True)
        daily = basin_mean(da).load()
    df = daily.to_dataframe(name="sst_c").reset_index()
    df["date"] = pd.to_datetime(df["time"])
    df[["date", "sst_c"]].to_csv(OUT / "oisst_daily_basin.csv", index=False)


def grid_trend_oisst() -> None:
    path = FILES["OISST v2.1"]
    if not path.exists():
        return

    with xr.open_dataset(path) as ds:
        da = find_sst(ds).squeeze(drop=True).sel(time=slice(TREND_START, None))
        da = da.where(strict_med_mask(da))
        annual = da.resample(time="YS").mean(skipna=True)

        years = annual.time.dt.year.astype(float)
        x = years - years.mean()
        slope_per_year = ((annual * x).sum("time") / (x ** 2).sum("time"))
        slope_per_decade = (slope_per_year * 10).rename("trend_c_per_decade").load()

    slope_per_decade.to_dataset().to_netcdf(OUT / "oisst_grid_trend.nc")


def main() -> None:
    frames = []
    for name, path in FILES.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"Processing {name}")
        frames.append(monthly_from_dataset(path, name))

    if not frames:
        raise FileNotFoundError("No input datasets found. Run 1_download_data.py first.")

    monthly = add_anomalies(pd.concat(frames, ignore_index=True))
    monthly.to_csv(OUT / "monthly_basin_sst.csv", index=False)

    annual = calculate_annual(monthly)
    annual.to_csv(OUT / "annual_basin_sst.csv", index=False)

    summaries = []
    for dataset, g in monthly.groupby("dataset"):
        result = ols_trend(g)
        result["dataset"] = dataset
        summaries.append(result)
    summary = pd.DataFrame(summaries)[
        ["dataset", "n_months", "trend_c_per_decade",
         "ci95_low", "ci95_high", "p_value", "r_squared"]
    ]
    summary.to_csv(OUT / "trend_summary.csv", index=False)
    print("\nTrend summary (1982 onward, monthly anomalies, HAC(12) errors):")
    print(summary.to_string(index=False))

    calculate_oisst_daily()
    grid_trend_oisst()


if __name__ == "__main__":
    main()