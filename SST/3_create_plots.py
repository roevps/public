# =============================================================================
# Script: 3_create_plots.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Rebuild the Mediterranean SST trend, July, seasonal, exceedance and map plots.
#
# INPUT
# -----------------------------------------------------------------------------
# data/processed/* — created by 2_calculate_trends.py
#
# OUTPUT
# -----------------------------------------------------------------------------
# plots/panel_a_annual_anomaly.png ... plots/panel_e_spatial_trend.png
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data" / "processed"
PLOTS = SCRIPT_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_annual() -> None:
    df = pd.read_csv(DATA / "annual_basin_sst.csv")
    plt.figure(figsize=(12, 5))
    for name, g in df.groupby("dataset"):
        g = g[g["year"] >= 1900]
        plt.plot(g["year"], g["sst_anomaly_c"], label=name, linewidth=1.5)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Year")
    plt.ylabel("SST anomaly (°C), 1982–2011")
    plt.title("Mediterranean annual sea-surface temperature anomaly")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS / "panel_a_annual_anomaly.png", dpi=220)
    plt.close()


def plot_july() -> None:
    df = pd.read_csv(DATA / "monthly_basin_sst.csv", parse_dates=["date"])
    july = df[df["date"].dt.month == 7].copy()
    # OISST is used because it provides the high-resolution observational series.
    g = july[july["dataset"] == "OISST v2.1"]
    plt.figure(figsize=(9, 4.5))
    plt.bar(g["date"].dt.year, g["sst_c"])
    baseline = g[(g["date"].dt.year >= 1982) & (g["date"].dt.year <= 2011)]["sst_c"].mean()
    plt.axhline(baseline, linestyle="--", linewidth=1, label="1982–2011 July mean")
    plt.xlabel("Year")
    plt.ylabel("July SST (°C)")
    plt.title("Mediterranean July SST — OISST v2.1")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "panel_b_july_sst.png", dpi=220)
    plt.close()


def plot_seasonal_cycle() -> None:
    df = pd.read_csv(DATA / "monthly_basin_sst.csv", parse_dates=["date"])
    g = df[df["dataset"] == "OISST v2.1"].copy()
    g["year"] = g["date"].dt.year
    g["month"] = g["date"].dt.month
    latest_year = int(g["year"].max())

    plt.figure(figsize=(8, 5))
    for year, y in g.groupby("year"):
        if year == latest_year:
            continue
        plt.plot(y["month"], y["sst_c"], alpha=0.16, linewidth=0.8)

    latest = g[g["year"] == latest_year]
    plt.plot(latest["month"], latest["sst_c"], marker="o", linewidth=2.2,
             label=f"{latest_year} to date")
    plt.xticks(range(1, 13), list("JFMAMJJASOND"))
    plt.xlabel("Month")
    plt.ylabel("Basin mean SST (°C)")
    plt.title("Mediterranean seasonal SST cycle — OISST v2.1")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "panel_c_seasonal_cycle.png", dpi=220)
    plt.close()


def daily_percentile_threshold(df: pd.DataFrame) -> pd.Series:
    """
    Calendar-day 90th percentile from 1982–2011.

    Uses a ±5-day moving calendar window to stabilize daily percentiles.
    Feb 29 is mapped to day 59.5-like behavior by dropping it from threshold
    construction and interpolating if needed.
    """
    x = df.copy()
    x = x[~((x["date"].dt.month == 2) & (x["date"].dt.day == 29))]
    x["doy"] = x["date"].dt.dayofyear
    # In leap years, shift dates after Feb 28 back by one for a 365-day calendar.
    leap_after_feb = x["date"].dt.is_leap_year & (x["date"].dt.month > 2)
    x.loc[leap_after_feb, "doy"] -= 1

    base = x[(x["date"] >= "1982-01-01") & (x["date"] <= "2011-12-31")]
    thresholds = {}
    for d in range(1, 366):
        circular_distance = np.minimum(
            np.abs(base["doy"] - d),
            365 - np.abs(base["doy"] - d),
        )
        vals = base.loc[circular_distance <= 5, "sst_c"]
        thresholds[d] = vals.quantile(0.90)
    return pd.Series(thresholds)


def plot_exceedance_days() -> None:
    df = pd.read_csv(DATA / "oisst_daily_basin.csv", parse_dates=["date"]).dropna()
    threshold = daily_percentile_threshold(df)

    x = df.copy()
    x["doy"] = x["date"].dt.dayofyear
    leap_after_feb = x["date"].dt.is_leap_year & (x["date"].dt.month > 2)
    x.loc[leap_after_feb, "doy"] -= 1
    x.loc[(x["date"].dt.month == 2) & (x["date"].dt.day == 29), "doy"] = 59
    x["threshold"] = x["doy"].map(threshold)
    x["above"] = x["sst_c"] > x["threshold"]
    x["year"] = x["date"].dt.year

    annual = x.groupby("year", as_index=False).agg(
        days_above_90th=("above", "sum"),
        observations=("above", "count"),
    )
    # Exclude incomplete latest year from direct full-year comparison.
    annual["complete_year"] = annual["observations"] >= 365

    plt.figure(figsize=(9, 4.5))
    plt.bar(annual["year"], annual["days_above_90th"])
    plt.xlabel("Year")
    plt.ylabel("Days above 1982–2011 daily 90th percentile")
    plt.title("Mediterranean warm-threshold exceedance days — OISST v2.1")
    plt.tight_layout()
    plt.savefig(PLOTS / "panel_d_days_above_90th.png", dpi=220)
    plt.close()

    annual.to_csv(DATA / "oisst_days_above_90th.csv", index=False)


def plot_map() -> None:
    ds = xr.open_dataset(DATA / "oisst_grid_trend.nc")
    da = ds["trend_c_per_decade"]

    plt.figure(figsize=(11, 5))
    da.plot(
        x="lon", y="lat",
        cbar_kwargs={"label": "SST trend (°C per decade)"},
        robust=True,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Mediterranean SST trend since 1982 — OISST v2.1")
    plt.tight_layout()
    plt.savefig(PLOTS / "panel_e_spatial_trend.png", dpi=220)
    plt.close()
    ds.close()


def main() -> None:
    plot_annual()
    plot_july()
    plot_seasonal_cycle()
    plot_exceedance_days()
    plot_map()
    print(f"Plots written to {PLOTS.resolve()}")


if __name__ == "__main__":
    main()
