"""Cross-validation-driven model selection.

Instead of hardcoding "use ETS for monthly retail data," freecast backtests
every candidate model on every series (rolling-origin cross-validation) and
selects per series by an explicit accuracy metric. This is the "expert
system" — encoded as a measurement, not a rule table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    ADIDA,
    IMAPA,
    TSB,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
    CrostonClassic,
    CrostonSBA,
)
from utilsforecast.losses import bias, mase, rmsse, smape

METRIC_FNS: dict[str, Any] = {
    "mase": mase,
    "rmsse": rmsse,
    "smape": smape,
    "bias": bias,
}
SCALED_METRICS = {"mase", "rmsse"}

DEFAULT_REGULAR_POOL = ("AutoETS", "AutoARIMA", "AutoTheta", "AutoCES")
DEFAULT_INTERMITTENT_POOL = ("CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA")


def default_regular_models(season_length: int) -> list[Any]:
    return [
        AutoETS(season_length=season_length, alias="AutoETS"),
        AutoARIMA(season_length=season_length, alias="AutoARIMA"),
        AutoTheta(season_length=season_length, alias="AutoTheta"),
        AutoCES(season_length=season_length, alias="AutoCES"),
    ]


def default_intermittent_models() -> list[Any]:
    return [
        CrostonClassic(alias="CrostonClassic"),
        CrostonSBA(alias="CrostonSBA"),
        TSB(alpha_d=0.1, alpha_p=0.1, alias="TSB"),
        ADIDA(alias="ADIDA"),
        IMAPA(alias="IMAPA"),
    ]


@dataclass
class SelectionResult:
    """Per-series chosen model plus the backtest accuracy table behind it."""

    best_model: pl.DataFrame
    """Columns: unique_id, model, <metric>."""

    cv_accuracy: pl.DataFrame
    """Long-format per-(unique_id, model) accuracy for every candidate model."""

    metric: str


def select_models(
    df: pl.DataFrame,
    *,
    h: int,
    freq: str | int,
    season_length: int,
    models: list[Any] | None = None,
    n_windows: int = 2,
    metric: str = "mase",
    n_jobs: int = -1,
) -> SelectionResult:
    """Backtest ``models`` on ``df`` via rolling-origin CV and pick a winner per series.

    Parameters
    ----------
    df: long-format (unique_id, ds, y) frame, already contract-validated.
    h: forecast horizon, also used as the CV step/test window size.
    freq: pandas-style frequency string (or integer step) for the series.
    season_length: seasonal period used by seasonal models and MASE/RMSSE scaling.
    models: candidate model instances; defaults to the regular ETS/ARIMA/Theta/CES pool.
    n_windows: number of rolling-origin CV windows to backtest across.
    metric: one of "mase", "rmsse", "smape", "bias" — lower is better for all of them.
    """
    if metric not in METRIC_FNS:
        raise ValueError(f"Unknown metric {metric!r}; choose one of {sorted(METRIC_FNS)}")

    pool = models if models is not None else default_regular_models(season_length)
    model_names = [getattr(m, "alias", type(m).__name__) for m in pool]

    min_len = df.group_by("unique_id").agg(pl.len().alias("n")).select(pl.col("n").min()).item()
    required = h * (n_windows + 1) + 1
    if min_len < required:
        n_windows = max(1, (min_len - h - 1) // h) if min_len > h + 1 else 1
        n_windows = max(1, min(n_windows, 2))

    sf = StatsForecast(models=pool, freq=freq, n_jobs=n_jobs)
    cv_df = sf.cross_validation(h=h, df=df, n_windows=n_windows)

    metric_fn = METRIC_FNS[metric]
    kwargs = (
        {"df": cv_df, "models": model_names, "train_df": df}
        if metric in SCALED_METRICS
        else {
            "df": cv_df,
            "models": model_names,
        }
    )
    if metric in SCALED_METRICS:
        kwargs["seasonality"] = season_length

    acc = metric_fn(**kwargs)
    acc_long = acc.unpivot(
        index="unique_id", on=model_names, variable_name="model", value_name=metric
    )

    # bias is signed (0 is ideal); every other supported metric is a
    # non-negative error where lower is strictly better.
    rank_col = acc_long[metric].abs() if metric == "bias" else acc_long[metric]
    acc_long = acc_long.with_columns(rank_col.alias("_rank"))
    best = (
        acc_long.sort(["unique_id", "_rank"])
        .group_by("unique_id", maintain_order=True)
        .first()
        .drop("_rank")
    )
    acc_long = acc_long.drop("_rank")

    return SelectionResult(best_model=best, cv_accuracy=acc_long, metric=metric)
