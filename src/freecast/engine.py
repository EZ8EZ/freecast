"""Orchestrates the full freecast pipeline: validate -> classify -> select -> forecast.

Exogenous regressors
---------------------
Any column in the input besides ``unique_id``, ``ds``, ``y`` is treated as an
exogenous regressor (e.g. a promo flag, price, a known holiday indicator).
Forecasting past the end of history needs *known future values* for those
columns — freecast can't invent your promo calendar — so ``ForecastEngine.run``
takes an optional ``X_df`` with the same regressor columns covering exactly
the ``h`` future periods per series.

Only AutoARIMA (of the regular-series pool) actually has a mechanism for
exogenous regressors; AutoETS/AutoTheta/AutoCES accept the extra columns
without erroring but silently ignore them (classical exponential smoothing
and Theta have no covariate concept). This isn't a special case freecast
codes around — it's the CV-based selection this whole engine is built on
working correctly: if a regressor is genuinely predictive, AutoARIMA
backtests better because it alone can use that information, and wins
selection on its own merits. Croston-family (intermittent) models ignore
extra columns the same way and never need ``X_df``. The optional t0
foundation-model candidate has no mechanism for *future*-known regressors
either, so it's excluded from selection whenever regressor columns are
present, rather than risk it winning on a coincidence and quietly dropping
a real promo/price effect.
"""

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

    def run(self, df: pl.DataFrame, X_df: pl.DataFrame | None = None) -> ForecastResult:
        """Run the pipeline. ``X_df`` supplies known future values for any
        exogenous regressor columns present in ``df`` (e.g. a planned promo
        calendar) — see the module docstring's "Exogenous regressors"
        section for what it must contain and which models actually use it.
        """
        clean_df, report = contract.validate(
            df, min_history=self.min_history, on_error=self.on_error
        )
        regressor_cols = [c for c in clean_df.columns if c not in ("unique_id", "ds", "y")]
        X_df = self._validate_regressors(clean_df, X_df, regressor_cols)

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

            # t0 has no mechanism for the future-known regressors X_df
            # carries (only historical covariates), so including it as a
            # candidate when regressors are in play would let it silently
            # ignore information AutoARIMA is genuinely using — skip it
            # rather than risk it winning selection on a coincidence and
            # quietly dropping the promo/price/etc. effect.
            t0_forecast = None
            if self.use_foundation_model and not regressor_cols:
                foundation_acc, t0_forecast = selection.evaluate_foundation_model(
                    regular_df,
                    h=self.h,
                    freq=self._freq_pandas,
                    season_length=self.season_length,
                    levels=tuple(self.levels),
                    metric=self.metric,
                )
                reg_sel = selection.merge_foundation_candidate(reg_sel, foundation_acc)

            reg_X_df = None
            if X_df is not None:
                reg_X_df = X_df.filter(
                    pl.col("unique_id").is_in(regular_df["unique_id"].unique().to_list())
                )

            selection_parts.append(reg_sel.best_model)
            forecast_parts.append(
                self._forecast_group(
                    regular_df,
                    reg_models,
                    reg_sel.best_model,
                    has_native_intervals=True,
                    extra_forecast=t0_forecast,
                    X_df=reg_X_df,
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

    def _validate_regressors(
        self, clean_df: pl.DataFrame, X_df: pl.DataFrame | None, regressor_cols: list[str]
    ) -> pl.DataFrame | None:
        if not regressor_cols:
            if X_df is not None:
                raise ValueError(
                    "X_df was provided but the input has no exogenous regressor columns "
                    "(everything besides unique_id, ds, y) to supply future values for."
                )
            return None

        if X_df is None:
            raise contract.ContractError(
                f"Input has exogenous regressor column(s) {regressor_cols}, but no X_df was "
                "given. Forecasting past the end of history needs known future values for "
                "them (e.g. a planned promo calendar) — freecast can't invent them. Pass "
                "X_df with unique_id, ds, and these columns, covering exactly the next h "
                "periods for every series."
            )

        X_df = contract.parse_ds_column(X_df)
        missing_cols = [c for c in ("unique_id", "ds", *regressor_cols) if c not in X_df.columns]
        if missing_cols:
            raise contract.ContractError(
                f"X_df is missing required column(s) {missing_cols}. It must carry unique_id, "
                f"ds, and every regressor column present in the input: {regressor_cols}."
            )

        counts = X_df.group_by("unique_id").agg(pl.len().alias("n"))
        wrong_length = counts.filter(pl.col("n") != self.h)
        if wrong_length.height > 0:
            raise contract.ContractError(
                f"X_df must have exactly h={self.h} rows per unique_id (one per future "
                f"period); {wrong_length.height} series don't, e.g. "
                f"{wrong_length['unique_id'].to_list()[:5]}."
            )

        input_ids = set(clean_df["unique_id"].unique().to_list())
        x_ids = set(X_df["unique_id"].unique().to_list())
        missing_ids = input_ids - x_ids
        if missing_ids:
            raise contract.ContractError(
                f"X_df is missing future regressor values for {len(missing_ids)} series in "
                f"the input, e.g. {sorted(missing_ids)[:5]}."
            )
        return X_df

    def _forecast_group(
        self,
        df: pl.DataFrame,
        models: list[Any],
        best_model: pl.DataFrame,
        *,
        has_native_intervals: bool,
        extra_forecast: pl.DataFrame | None = None,
        X_df: pl.DataFrame | None = None,
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
            wide = sf.forecast(h=self.h, df=df, level=self.levels, X_df=X_df)
        else:
            n_windows = max(min(max(self.n_windows, 2), max_feasible_windows), 2)
            ci = build_conformal_intervals(h=self.h, n_windows=n_windows)
            wide = sf.forecast(
                h=self.h, df=df, level=self.levels, prediction_intervals=ci, X_df=X_df
            )
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
