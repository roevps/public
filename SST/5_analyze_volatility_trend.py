# =============================================================================
# Script: 5_analyze_volatility_trend.py
# Version: 1.1
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-10
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Test whether GARCH conditional SST volatility has increased since 1982.
#
# INPUT
# -----------------------------------------------------------------------------
# data/processed/garch_conditional_volatility.csv — created by 4_garch_model.py
#
# OUTPUT
# -----------------------------------------------------------------------------
# data/processed/garch_volatility_trend_summary.csv
# plots/garch_volatility_trend_<dataset>.png, plots/garch_volatility_trend_summary.png
# =============================================================================

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data" / "processed"
PLOTS = SCRIPT_DIR / "plots"

INPUT = DATA / "garch_conditional_volatility.csv"
OUTPUT = DATA / "garch_volatility_trend_summary.csv"

START = "1982-01-01"
HAC_LAGS = 12
WINDOW_MONTHS = 120


def safe_name(text: str) -> str:
    """Return a filename-safe dataset name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def year_decimal(dates: pd.Series) -> np.ndarray:
    """Convert timestamps to decimal years for trend regression."""
    return (
        dates.dt.year
        + (dates.dt.dayofyear - 1)
        / np.where(dates.dt.is_leap_year, 366, 365)
    ).to_numpy(dtype=float)


def fit_volatility_trend(g: pd.DataFrame) -> dict:
    """Fit a linear HAC trend to monthly GARCH conditional volatility."""
    g = (
        g[g["date"] >= START]
        .dropna(subset=["conditional_volatility_c"])
        .sort_values("date")
        .copy()
    )

    if len(g) < 120:
        raise ValueError(f"Only {len(g)} observations; at least 120 are required.")

    x = year_decimal(g["date"])
    X = sm.add_constant(x)
    y = g["conditional_volatility_c"].to_numpy(dtype=float)

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS},
    )

    slope_per_decade = model.params[1] * 10.0
    ci = model.conf_int(alpha=0.05)[1] * 10.0

    first_n = min(WINDOW_MONTHS, len(g))
    last_n = min(WINDOW_MONTHS, len(g))

    first_mean = g["conditional_volatility_c"].iloc[:first_n].mean()
    last_mean = g["conditional_volatility_c"].iloc[-last_n:].mean()
    difference = last_mean - first_mean
    percent_change = (
        difference / first_mean * 100.0
        if first_mean != 0
        else np.nan
    )

    return {
        "n_months": len(g),
        "start_date": g["date"].iloc[0].date().isoformat(),
        "end_date": g["date"].iloc[-1].date().isoformat(),
        "trend_c_per_decade": slope_per_decade,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
        "mean_first_120m_c": first_mean,
        "mean_last_120m_c": last_mean,
        "last_minus_first_c": difference,
        "change_percent": percent_change,
        "_model": model,
        "_x": x,
        "_y": y,
        "_dates": g["date"].to_numpy(),
    }


def plot_volatility_trend(dataset: str, result: dict) -> None:
    """Plot monthly conditional volatility, 12-month mean and linear trend."""
    dates = pd.to_datetime(result["_dates"])
    y = result["_y"]
    x = result["_x"]
    model = result["_model"]

    rolling = (
        pd.Series(y, index=dates)
        .rolling(12, min_periods=6, center=True)
        .mean()
    )
    fitted = model.predict(sm.add_constant(x))

    plt.figure(figsize=(11, 5))
    plt.plot(dates, y, linewidth=0.7, alpha=0.35, label="Monthly GARCH volatility")
    plt.plot(
        rolling.index,
        rolling.values,
        linewidth=1.8,
        label="12-month rolling mean",
    )
    plt.plot(
        dates,
        fitted,
        linewidth=2.0,
        linestyle="--",
        label="Linear volatility trend",
    )

    slope = result["trend_c_per_decade"]
    p_value = result["p_value"]

    plt.xlabel("Date")
    plt.ylabel("Conditional volatility, σ(t) (°C)")
    plt.title(
        f"{dataset}: GARCH conditional SST volatility\n"
        f"Trend = {slope:+.4f} °C/decade, p = {p_value:.3g}"
    )
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        PLOTS / f"garch_volatility_trend_{safe_name(dataset)}.png",
        dpi=220,
    )
    plt.close()


def plot_summary(results: dict[str, dict]) -> None:
    """Plot the 12-month rolling GARCH volatility for all SST products."""
    plt.figure(figsize=(12, 6))

    significant = 0

    for dataset, result in results.items():
        dates = pd.to_datetime(result["_dates"])
        y = result["_y"]

        rolling = (
            pd.Series(y, index=dates)
            .rolling(12, min_periods=6, center=True)
            .mean()
        )

        slope = result["trend_c_per_decade"]
        p_value = result["p_value"]

        if p_value < 0.05:
            significant += 1

        plt.plot(
            rolling.index,
            rolling.values,
            linewidth=1.7,
            label=(
                f"{dataset} "
                f"({slope:+.4f} °C/decade, p={p_value:.3f})"
            ),
        )

    if significant == 0:
        subtitle = "No dataset shows a statistically significant linear volatility trend"
    elif significant == 1:
        subtitle = "1 dataset shows a statistically significant linear volatility trend"
    else:
        subtitle = (
            f"{significant} datasets show statistically significant "
            "linear volatility trends"
        )

    plt.xlabel("Date")
    plt.ylabel("12-month mean conditional volatility, σ(t) (°C)")
    plt.title(
        "Mediterranean GARCH conditional SST volatility — all datasets\n"
        + subtitle
    )
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(
        PLOTS / "garch_volatility_trend_summary.png",
        dpi=220,
    )
    plt.close()


def interpretation(row: pd.Series) -> str:
    """Return a short statistical interpretation of the volatility trend."""
    slope = row["trend_c_per_decade"]
    p = row["p_value"]

    if p < 0.05 and slope > 0:
        return "significant increase"
    if p < 0.05 and slope < 0:
        return "significant decrease"
    return "no significant linear trend"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found.\n"
            "Run 4_garch_model.py first and make sure it writes to "
            f"{DATA}."
        )

    df = pd.read_csv(INPUT, parse_dates=["date"])

    required = {
        "dataset",
        "date",
        "conditional_volatility_c",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}; "
            f"available={list(df.columns)}"
        )

    rows = []
    results = {}

    for dataset, g in df.groupby("dataset", sort=True):
        result = fit_volatility_trend(g)
        results[dataset] = result
        plot_volatility_trend(dataset, result)

        rows.append(
            {
                **{
                    key: value
                    for key, value in result.items()
                    if not key.startswith("_")
                },
                "dataset": dataset,
            }
        )

    summary = pd.DataFrame(rows)[
        [
            "dataset",
            "n_months",
            "start_date",
            "end_date",
            "trend_c_per_decade",
            "ci95_low",
            "ci95_high",
            "p_value",
            "r_squared",
            "mean_first_120m_c",
            "mean_last_120m_c",
            "last_minus_first_c",
            "change_percent",
        ]
    ]

    summary["interpretation"] = summary.apply(interpretation, axis=1)
    summary.to_csv(OUTPUT, index=False)

    plot_summary(results)

    print("\nGARCH conditional-volatility trend:")
    print(
        summary[
            [
                "dataset",
                "trend_c_per_decade",
                "ci95_low",
                "ci95_high",
                "p_value",
                "mean_first_120m_c",
                "mean_last_120m_c",
                "change_percent",
                "interpretation",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved summary: {OUTPUT}")
    print(f"Saved plots:   {PLOTS}")
    print(
        "Summary plot: "
        f"{PLOTS / 'garch_volatility_trend_summary.png'}"
    )


if __name__ == "__main__":
    main()