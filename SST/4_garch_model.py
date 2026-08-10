# =============================================================================
# Script: 4_garch_model.py
# Version: 1.0
# Authors: Roland
# AI pair: GPT-5.6 Sol
# Python: 3.12+
# Date: 2026-08-08
#
# DESCRIPTION
# -----------------------------------------------------------------------------
# Fit GARCH(1,1) models to detrended monthly Mediterranean SST anomalies.
#
# INPUT
# -----------------------------------------------------------------------------
# data/processed/monthly_basin_sst.csv — created by 2_calculate_trends.py
#
# OUTPUT
# -----------------------------------------------------------------------------
# data/processed/garch_parameters.csv, garch_conditional_volatility.csv,
# plots/garch_<dataset>.png, data/processed/garch_<dataset>_summary.txt
# =============================================================================

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data" / "processed"
PLOTS = SCRIPT_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

START = "1982-01-01"


def safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def detrend(g: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a linear trend from already de-seasonalized monthly anomalies.

    GARCH is then applied to residual variability rather than to the deterministic
    warming trend itself.
    """
    g = g.sort_values("date").dropna(subset=["sst_anomaly_c"]).copy()
    t = np.arange(len(g), dtype=float)
    X = sm.add_constant(t)
    fit = sm.OLS(g["sst_anomaly_c"].to_numpy(), X).fit()
    g["trend_component_c"] = fit.predict(X)
    g["residual_c"] = g["sst_anomaly_c"] - g["trend_component_c"]
    return g


def main() -> None:
    df = pd.read_csv(DATA / "monthly_basin_sst.csv", parse_dates=["date"])
    df = df[df["date"] >= START].copy()

    parameter_rows = []
    volatility_rows = []

    for dataset, g0 in df.groupby("dataset"):
        g = detrend(g0)
        if len(g) < 120:
            print(f"Skipping {dataset}: only {len(g)} observations")
            continue

        # Scaling by 100 improves numerical conditioning; units become centi-°C.
        y = g["residual_c"].to_numpy() * 100.0

        # Mean is zero because anomalies were de-seasonalized and then detrended.
        model = arch_model(
            y,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="StudentsT",
            rescale=False,
        )
        result = model.fit(disp="off")

        # ARCH-LM before and after standardization.
        lm_before = het_arch(y, nlags=12)
        std_resid = pd.Series(result.std_resid).dropna().to_numpy()
        lm_after = het_arch(std_resid, nlags=12)

        params = result.params.to_dict()
        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        persistence = alpha + beta

        row = {
            "dataset": dataset,
            "n": len(y),
            "loglikelihood": result.loglikelihood,
            "aic": result.aic,
            "bic": result.bic,
            "omega": params.get("omega", np.nan),
            "alpha_1": alpha,
            "beta_1": beta,
            "nu_student_t": params.get("nu", np.nan),
            "persistence_alpha_plus_beta": persistence,
            "arch_lm_p_before": lm_before[1],
            "arch_lm_p_after": lm_after[1],
        }
        parameter_rows.append(row)

        vol_c = np.asarray(result.conditional_volatility) / 100.0
        vol_df = pd.DataFrame({
            "dataset": dataset,
            "date": g["date"].to_numpy(),
            "residual_c": g["residual_c"].to_numpy(),
            "conditional_volatility_c": vol_c,
        })
        volatility_rows.append(vol_df)

        stem = safe_name(dataset)
        (DATA / f"garch_{stem}_summary.txt").write_text(
            result.summary().as_text(), encoding="utf-8"
        )

        plt.figure(figsize=(11, 5))
        plt.plot(g["date"], g["residual_c"], linewidth=0.9, label="Detrended anomaly")
        plt.plot(g["date"], vol_c, linewidth=1.3, label="GARCH σ(t)")
        plt.plot(g["date"], -vol_c, linewidth=1.3)
        plt.axhline(0, linewidth=0.7)
        plt.xlabel("Date")
        plt.ylabel("°C")
        plt.title(f"{dataset}: GARCH(1,1) conditional volatility")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(PLOTS / f"garch_{stem}.png", dpi=220)
        plt.close()

    if not parameter_rows:
        raise RuntimeError("No dataset had enough observations for GARCH.")

    params_df = pd.DataFrame(parameter_rows)
    params_df.to_csv(DATA / "garch_parameters.csv", index=False)
    pd.concat(volatility_rows, ignore_index=True).to_csv(
        DATA / "garch_conditional_volatility.csv", index=False
    )

    print("\nGARCH(1,1) results:")
    print(params_df.to_string(index=False))


if __name__ == "__main__":
    main()
