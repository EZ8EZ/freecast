"""Orchestrates the full freecast pipeline: validate -> classify -> select -> forecast."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import polars as pl
from statsforecast import StatsForecast

from freecast import contract, intermittent, selection
from freecast.intervals import DEFAULT_LEVELS, build_conformal_intervals

_FREQ_SEASON_LENGTH = {
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "MS": 12,
    "Q": 4,
    "QS": 4,
    "Y": 1,
    "A": 1,
    "AS": 1,
    "H": 24,
}


def infer_season_length(freq: str | int) -> int:
    """Best-effort seasonal period for a pandas-style frequency string."""
    if isinstance(freq, int):
        return 1
    key = re.sub(r"^\d+", "", freq.upper())
    return _FREQ_SEASON_LENGTH.get(key, 1)


@dataclass
class ForecastResult:
    forecasts: pl.DataFrame
    """unique_id, ds, model, y, plus lo-<level>/hi-<level> per requested level."""

    selection: pl.DataFrame
    """unique_id, model, <metric>: the chosen model and its backtested accuracy."""

    classification: pl.DataFrame
    """unique_id, adi, cv2, category: intermittent-demand routing decision."""

    validation: contract.ValidationReport


class ForecastEngine:
    """Batch forecasting engine: one call handles thousands of independent series."""

    def __init__(
        self,
        *,
        h: int,
        freq: str | int,
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        metric: str = "mase",
        n_windows: int = 2,
        min_history: int = 6,
        on_error: str = "raise",
        n_jobs: int = -1,
    ) -> None:
        self.h = h
        self.freq = freq
        self.levels = list(levels)
        self.metric = metric
        self.n_windows = n_windows
        self.min_history = min_history
        self.on_error = on_error
        self.n_jobs = n_jobs
        self.season_length = infer_season_length(freq)

    def run(self, df: pl.DataFrame) -> ForecastResult:
        clean_df, report = contract.validate(
            df, min_history=self.min_history, on_error=self.on_error
        )

        classification = intermittent.classify_series(clean_df)
        regular_df, intermittent_df = intermittent.split_by_demand_type(clean_df, classification)

        selection_parts: list[pl.DataFrame] = []
        forecast_parts: list[pl.DataFrame] = []

        if regular_df.height > 0:
            reg_models = selection.default_regular_models(self.season_length)
            reg_sel = selection.select_models(
                regular_df,
                h=self.h,
                freq=self.freq,
                season_length=self.season_length,
                models=reg_models,
                n_windows=self.n_windows,
                metric=self.metric,
                n_jobs=self.n_jobs,
            )
            selection_parts.append(reg_sel.best_model)
            forecast_parts.append(
                self._forecast_group(
                    regular_df, reg_models, reg_sel.best_model, has_native_intervals=True
                )
            )

        if intermittent_df.height > 0:
            int_models = selection.default_intermittent_models()
            int_sel = selection.select_models(
                intermittent_df,
                h=self.h,
                freq=self.freq,
                season_length=self.season_length,
                models=int_models,
                n_windows=self.n_windows,
                metric=self.metric,
                n_jobs=self.n_jobs,
            )
            selection_parts.append(int_sel.best_model)
            forecast_parts.append(
                self._forecast_group(
                    intermittent_df, int_models, int_sel.best_model, has_native_intervals=False
                )
            )

        selection_df = pl.concat(selection_parts, how="vertical")
        forecasts_df = pl.concat(forecast_parts, how="vertical")

        return ForecastResult(
            forecasts=forecasts_df,
            selection=selection_df,
            classification=classification,
            validation=report,
        )

    def _forecast_group(
        self,
        df: pl.DataFrame,
        models: list[Any],
        best_model: pl.DataFrame,
        *,
        has_native_intervals: bool,
    ) -> pl.DataFrame:
        model_names = [getattr(m, "alias", type(m).__name__) for m in models]
        sf = StatsForecast(models=models, freq=self.freq, n_jobs=self.n_jobs)

        # Conformal intervals need >= 2 full backtest windows of length h left
        # over after fitting; on very short series (some M3 Yearly series
        # have as few as 14 training points at h=6) that leaves nothing to
        # fit on. ETS/ARIMA/Theta/CES have their own parametric interval as a
        # fallback in that case. Croston-family models have no such
        # fallback — they only know how to produce intervals conformally —
        # but they tolerate short series fine, so they always go the
        # conformal route.
        min_len = df.group_by("unique_id").agg(pl.len().alias("n")).select(pl.col("n").min()).item()
        max_feasible_windows = (min_len - 1) // self.h - 1
        if has_native_intervals and max_feasible_windows < 2:
            wide = sf.forecast(h=self.h, df=df, level=self.levels)
        else:
            n_windows = max(min(max(self.n_windows, 2), max_feasible_windows), 2)
            ci = build_conformal_intervals(h=self.h, n_windows=n_windows)
            wide = sf.forecast(h=self.h, df=df, level=self.levels, prediction_intervals=ci)
        wide = wide.join(best_model.select(["unique_id", "model"]), on="unique_id", how="left")

        y_col = pl.coalesce(
            [pl.when(pl.col("model") == name).then(pl.col(name)) for name in model_names]
        ).alias("y_hat")

        select_exprs = [pl.col("unique_id"), pl.col("ds"), pl.col("model"), y_col]
        for level in self.levels:
            lo_col = pl.coalesce(
                [
                    pl.when(pl.col("model") == name).then(pl.col(f"{name}-lo-{level}"))
                    for name in model_names
                ]
            ).alias(f"lo-{level}")
            hi_col = pl.coalesce(
                [
                    pl.when(pl.col("model") == name).then(pl.col(f"{name}-hi-{level}"))
                    for name in model_names
                ]
            ).alias(f"hi-{level}")
            select_exprs.extend([lo_col, hi_col])

        return wide.select(select_exprs)
