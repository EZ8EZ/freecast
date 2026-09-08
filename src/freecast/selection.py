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
    freq: Polars-style frequency string (or integer step) for the series —
        i.e. ``ResolvedFreq.polars`` from ``freecast.freq.resolve_freq``.
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

    best = _pick_best(acc_long, metric)
    return SelectionResult(best_model=best, cv_accuracy=acc_long, metric=metric)


def _pick_best(acc_long: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Pick the winning model per series from a long (unique_id, model, <metric>) table.

    bias is signed (0 is ideal); every other supported metric is a
    non-negative error where lower is strictly better.
    """
    rank_col = acc_long[metric].abs() if metric == "bias" else acc_long[metric]
    ranked = acc_long.with_columns(rank_col.alias("_rank"))
    return (
        ranked.sort(["unique_id", "_rank"])
        .group_by("unique_id", maintain_order=True)
        .first()
        .drop("_rank")
    )


def evaluate_foundation_model(
    df: pl.DataFrame,
    *,
    h: int,
    freq: str | int,
    season_length: int,
    levels: tuple[int, ...] = (80,),
    metric: str = "mase",
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Backtest and forecast with the optional t0 foundation model (see ``foundation.py``).

    Zero-shot models don't need per-series training, so unlike
    ``select_models`` this uses a single held-out window (the last ``h``
    points) rather than multiple rolling-origin windows — cheap and
    sufficient to compare against the statistical pool's CV accuracy.

    Returns ``(accuracy, forecast)`` where ``accuracy`` is a long
    (unique_id, model="T0", <metric>) table (empty if t0 is unavailable or
    the requested levels aren't supported), and ``forecast`` is the
    corresponding wide forecast frame (or None if accuracy is empty).
    """
    from freecast.foundation import (
        MODEL_NAME,
        FoundationModelUnavailable,
        T0Model,
        levels_supported,
    )

    empty = pl.DataFrame(schema={"unique_id": pl.Utf8, "model": pl.Utf8, metric: pl.Float64})
    if not levels_supported(levels):
        return empty, None

    min_len = df.group_by("unique_id").agg(pl.len().alias("n")).select(pl.col("n").min()).item()
    if min_len <= h:
        return empty, None

    try:
        model = T0Model.load()
    except FoundationModelUnavailable:
        return empty, None

    train_parts, test_parts = [], []
    for _, series in df.sort(["unique_id", "ds"]).group_by("unique_id"):
        train_parts.append(series.head(series.height - h))
        test_parts.append(series.tail(h))
    train_df = pl.concat(train_parts)
    test_df = pl.concat(test_parts)

    backtest_wide = model.predict(train_df, h=h, freq=freq, levels=levels)
    joined = backtest_wide.join(test_df.select(["unique_id", "ds", "y"]), on=["unique_id", "ds"])

    metric_fn = METRIC_FNS[metric]
    kwargs = (
        {"df": joined, "models": [MODEL_NAME], "train_df": train_df}
        if metric in SCALED_METRICS
        else {"df": joined, "models": [MODEL_NAME]}
    )
    if metric in SCALED_METRICS:
        kwargs["seasonality"] = season_length
    acc = metric_fn(**kwargs)
    acc_long = acc.unpivot(
        index="unique_id", on=[MODEL_NAME], variable_name="model", value_name=metric
    )

    forecast_wide = model.predict(df, h=h, freq=freq, levels=levels)
    return acc_long, forecast_wide


def merge_foundation_candidate(
    result: SelectionResult, foundation_acc: pl.DataFrame
) -> SelectionResult:
    """Fold the optional t0 foundation-model accuracy into an existing selection result."""
    if foundation_acc is None or foundation_acc.height == 0:
        return result
    combined = pl.concat([result.cv_accuracy, foundation_acc], how="vertical")
    best = _pick_best(combined, result.metric)
    return SelectionResult(best_model=best, cv_accuracy=combined, metric=result.metric)
