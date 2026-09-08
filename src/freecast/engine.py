"""Orchestrates the full freecast pipeline: validate -> classify -> select -> forecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from statsforecast import StatsForecast

from freecast import contract, intermittent, selection
from freecast.freq import resolve_freq
from freecast.intervals import DEFAULT_LEVELS, build_conformal_intervals


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
        season_length: int | None = None,
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        metric: str = "mase",
        n_windows: int = 2,
        min_history: int = 6,
        on_error: str = "raise",
        n_jobs: int = -1,
        use_foundation_model: bool = False,
    ) -> None:
        """
        season_length: override the seasonal period used by seasonal models
            and MASE/RMSSE scaling. By default this is inferred from ``freq``
            (e.g. 12 for monthly data), which is right for genuine calendar
            data but wrong when ``ds`` carries a frequency label with no real
            periodicity behind it (e.g. synthetic daily dates standing in for
            an arbitrarily-ordered sequence) — pass an explicit value in that
            case rather than letting the freq default inject a seasonal
            pattern that doesn't exist in the data.
        """
        self.h = h
        self.freq = freq
        resolved = resolve_freq(freq)
        self._freq_polars = resolved.polars
        self._freq_pandas = resolved.pandas
        self.levels = list(levels)
        self.metric = metric
        self.n_windows = n_windows
        self.min_history = min_history
        self.on_error = on_error
        self.n_jobs = n_jobs
        self.season_length = season_length if season_length is not None else resolved.season_length
        self.use_foundation_model = use_foundation_model

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
                freq=self._freq_polars,
                season_length=self.season_length,
                models=reg_models,
                n_windows=self.n_windows,
                metric=self.metric,
                n_jobs=self.n_jobs,
            )

            t0_forecast = None
            if self.use_foundation_model:
                foundation_acc, t0_forecast = selection.evaluate_foundation_model(
                    regular_df,
                    h=self.h,
                    freq=self._freq_pandas,
                    season_length=self.season_length,
                    levels=tuple(self.levels),
                    metric=self.metric,
                )
                reg_sel = selection.merge_foundation_candidate(reg_sel, foundation_acc)

            selection_parts.append(reg_sel.best_model)
            forecast_parts.append(
                self._forecast_group(
                    regular_df,
                    reg_models,
                    reg_sel.best_model,
                    has_native_intervals=True,
                    extra_forecast=t0_forecast,
                )
            )

        if intermittent_df.height > 0:
            int_models = selection.default_intermittent_models()
            int_sel = selection.select_models(
                intermittent_df,
                h=self.h,
                freq=self._freq_polars,
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
        extra_forecast: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        model_names = [getattr(m, "alias", type(m).__name__) for m in models]
        sf = StatsForecast(models=models, freq=self._freq_polars, n_jobs=self.n_jobs)

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
        if extra_forecast is not None:
            wide = wide.join(extra_forecast, on=["unique_id", "ds"], how="full", coalesce=True)
            model_names = [*model_names, "T0"]

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
