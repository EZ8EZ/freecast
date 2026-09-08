"""Forecast Value Added (FVA): does each layer of the process actually help?

Standard practice (Gilliland, *Business Forecasting: Practical Problems and
Solutions*): compare the accuracy of a naive baseline, the statistical
forecast, and any planner override against realized actuals.

    FVA = baseline_error - challenger_error

Positive FVA means the challenger beat the baseline; negative means it made
things worse. Published research on manual overrides finds roughly 40-60%
of them destroy accuracy rather than improve it — FVA is how a team finds
out which ones, instead of assuming judgment always helps, and is the
standard evidence base for retiring an override layer that isn't earning
its keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

SUPPORTED_METRICS = ("mae", "mape")


@dataclass
class FVAResult:
    per_series: pl.DataFrame
    """unique_id, naive_error, statistical_error, statistical_fva, and
    (if overrides were supplied) override_error, override_fva."""

    overall: dict[str, float]
    """Same columns as per_series, averaged across series."""

    metric: str


def _score(df: pl.DataFrame, yhat_col: str, metric: str, out_col: str) -> pl.DataFrame:
    if metric == "mae":
        err = (pl.col("y") - pl.col(yhat_col)).abs()
    else:  # mape
        err = (pl.col("y") - pl.col(yhat_col)).abs() / pl.col("y").abs()
    return (
        df.with_columns(err.alias("_err"))
        .group_by("unique_id")
        .agg(pl.col("_err").mean().alias(out_col))
    )


def compute_fva(
    train_df: pl.DataFrame,
    forecasts: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    freq: str | int,
    season_length: int = 1,
    overrides: pl.DataFrame | None = None,
    metric: str = "mae",
) -> FVAResult:
    """Compute per-series and overall Forecast Value Added.

    Parameters
    ----------
    train_df: long-format (unique_id, ds, y) history used to fit the naive baseline.
    forecasts: (unique_id, ds, y_hat) — the statistical forecast being evaluated.
    actuals: (unique_id, ds, y) — realized values for the forecast period.
    freq: pandas-style frequency string (or integer step).
    season_length: seasonal period for the naive baseline; 1 uses plain Naive,
        otherwise SeasonalNaive.
    overrides: optional (unique_id, ds, override_y_hat) — e.g. from
        ``OverrideStore.active_overrides()`` — to also score the override layer.
    metric: "mae" or "mape".
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose one of {SUPPORTED_METRICS}")

    h = forecasts.group_by("unique_id").agg(pl.len().alias("n")).select(pl.col("n").max()).item()
    naive_model = (
        Naive(alias="naive")
        if season_length <= 1
        else SeasonalNaive(season_length=season_length, alias="naive")
    )
    sf = StatsForecast(models=[naive_model], freq=freq, n_jobs=-1)
    naive_wide = sf.forecast(h=h, df=train_df).rename({"naive": "naive_y_hat"})

    eval_df = actuals.join(
        forecasts.select(["unique_id", "ds", pl.col("y_hat").alias("statistical_y_hat")]),
        on=["unique_id", "ds"],
    ).join(naive_wide.select(["unique_id", "ds", "naive_y_hat"]), on=["unique_id", "ds"])

    naive_err = _score(eval_df, "naive_y_hat", metric, "naive_error")
    stat_err = _score(eval_df, "statistical_y_hat", metric, "statistical_error")
    per_series = naive_err.join(stat_err, on="unique_id").with_columns(
        (pl.col("naive_error") - pl.col("statistical_error")).alias("statistical_fva")
    )

    if overrides is not None and overrides.height > 0:
        eval_df = eval_df.join(
            overrides.select(["unique_id", "ds", "override_y_hat"]),
            on=["unique_id", "ds"],
            how="inner",
        )
        override_err = _score(eval_df, "override_y_hat", metric, "override_error")
        per_series = per_series.join(override_err, on="unique_id", how="left").with_columns(
            (pl.col("statistical_error") - pl.col("override_error")).alias("override_fva")
        )

    numeric_cols = [c for c in per_series.columns if c != "unique_id"]
    overall = {
        c: float(v)  # type: ignore[arg-type]  # these columns are always float64
        for c in numeric_cols
        if (v := per_series[c].mean()) is not None
    }

    return FVAResult(per_series=per_series, overall=overall, metric=metric)
