from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from freecast import selection
from freecast.contract import ContractError
from freecast.engine import ForecastEngine


def _future_months(start: datetime.date, n: int) -> list[datetime.date]:
    dates = []
    y, m = start.year, start.month
    for _ in range(n):
        m += 1
        if m > 12:
            m = 1
            y += 1
        dates.append(datetime.date(y, m, 1))
    return dates


def test_engine_season_length_agrees_across_freq_dialects():
    # A Polars-style freq ("1mo") and its pandas-style equivalent ("MS") must
    # resolve to the same season length — this used to silently diverge
    # (Polars-style strings fell through to season_length=1), which meant
    # every M3 benchmark group (fed Polars-style freqs by freecast/bench/runner.py)
    # was fit with no seasonality at all.
    assert ForecastEngine(h=6, freq="1mo", n_windows=1).season_length == 12
    assert ForecastEngine(h=6, freq="MS", n_windows=1).season_length == 12
    assert ForecastEngine(h=6, freq="1d", n_windows=1).season_length == 7
    assert ForecastEngine(h=6, freq="D", n_windows=1).season_length == 7
    assert ForecastEngine(h=6, freq="1q", n_windows=1).season_length == 4
    assert ForecastEngine(h=6, freq="Q", n_windows=1).season_length == 4
    assert ForecastEngine(h=6, freq=1, n_windows=1).season_length == 1


def test_engine_end_to_end_mixed(mixed_series_df):
    engine = ForecastEngine(h=6, freq="1mo", n_windows=1)
    result = engine.run(mixed_series_df)

    all_ids = set(mixed_series_df["unique_id"].unique().to_list())
    assert set(result.selection["unique_id"].to_list()) == all_ids
    assert set(result.classification["unique_id"].to_list()) == all_ids
    assert set(result.forecasts["unique_id"].unique().to_list()) == all_ids

    for uid in all_ids:
        n_rows = result.forecasts.filter(result.forecasts["unique_id"] == uid).height
        assert n_rows == 6

    assert "lo-80" in result.forecasts.columns
    assert "hi-80" in result.forecasts.columns
    assert "lo-95" in result.forecasts.columns
    assert "hi-95" in result.forecasts.columns
    lo_le_hi = (result.forecasts["lo-80"] <= result.forecasts["hi-80"]).all()
    assert lo_le_hi


def test_engine_regular_only(regular_series_df):
    engine = ForecastEngine(h=6, freq="1mo", n_windows=1)
    result = engine.run(regular_series_df)
    assert (result.classification["category"] == "smooth").all()
    assert result.forecasts.height == 4 * 6


def test_engine_foundation_model_unavailable_degrades_gracefully(regular_series_df, monkeypatch):
    """use_foundation_model=True must not break the run when t0 can't be loaded."""

    def fake_empty(*_args, **_kwargs):
        return (
            pl.DataFrame(schema={"unique_id": pl.Utf8, "model": pl.Utf8, "mase": pl.Float64}),
            None,
        )

    monkeypatch.setattr(selection, "evaluate_foundation_model", fake_empty)

    engine = ForecastEngine(h=6, freq="1mo", n_windows=1, use_foundation_model=True)
    result = engine.run(regular_series_df)
    assert result.forecasts.height == 4 * 6
    assert "T0" not in result.selection["model"].to_list()


def test_engine_foundation_model_can_win(regular_series_df, monkeypatch):
    """When t0 backtests better than every statistical model, its forecast is used."""
    uids = regular_series_df["unique_id"].unique().to_list()
    future_dates = _future_months(regular_series_df["ds"].max(), 6)

    def fake_foundation(df, *, h, freq, season_length, levels, metric):
        acc = pl.DataFrame(
            {"unique_id": uids, "model": ["T0"] * len(uids), metric: [0.0] * len(uids)}
        )
        rows = []
        for uid in uids:
            for ds in future_dates[:h]:
                rows.append(
                    {"unique_id": uid, "ds": ds, "T0": 999.0, "T0-lo-80": 990.0, "T0-hi-80": 1010.0}
                )
        forecast = pl.DataFrame(rows).with_columns(pl.col("ds").cast(df.schema["ds"]))
        return acc, forecast

    monkeypatch.setattr(selection, "evaluate_foundation_model", fake_foundation)

    engine = ForecastEngine(h=6, freq="1mo", n_windows=1, levels=(80,), use_foundation_model=True)
    result = engine.run(regular_series_df)

    assert set(result.selection["model"].unique().to_list()) == {"T0"}
    assert (result.forecasts["y_hat"] == 999.0).all()
    assert (result.forecasts["lo-80"] == 990.0).all()
    assert (result.forecasts["hi-80"] == 1010.0).all()


def _promo_series_df(n: int = 60, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = []
    y, m = 2019, 1
    for _ in range(n):
        dates.append(datetime.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    promo = (rng.random(n) > 0.75).astype(float)
    y_vals = 100 + 10 * np.sin(np.arange(n) / 12 * 2 * np.pi) + 25 * promo + rng.normal(0, 2, n)
    return pl.DataFrame({"unique_id": ["sku1"] * n, "ds": dates, "y": y_vals, "promo": promo})


def test_engine_uses_exogenous_regressor(regular_series_df):
    df = _promo_series_df()
    h = 6
    future_dates = _future_months(df["ds"].max(), h)
    future_promo = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    X_df = pl.DataFrame({"unique_id": ["sku1"] * h, "ds": future_dates, "promo": future_promo})

    engine = ForecastEngine(h=h, freq="MS", n_windows=1)
    result = engine.run(df, X_df=X_df)

    assert result.selection.row(0, named=True)["model"] == "AutoARIMA"
    forecasts = result.forecasts.sort("ds")
    promo_rows = forecasts.filter(pl.Series(future_promo) == 1.0)
    no_promo_rows = forecasts.filter(pl.Series(future_promo) == 0.0)
    assert promo_rows["y_hat"].mean() > no_promo_rows["y_hat"].mean() + 10


def test_engine_requires_x_df_when_regressors_present():
    df = _promo_series_df()
    engine = ForecastEngine(h=6, freq="MS", n_windows=1)
    with pytest.raises(ContractError, match="exogenous regressor"):
        engine.run(df)


def test_engine_rejects_x_df_with_wrong_row_count():
    df = _promo_series_df()
    bad_x = pl.DataFrame(
        {
            "unique_id": ["sku1"] * 3,
            "ds": _future_months(df["ds"].max(), 3),
            "promo": [0.0, 1.0, 0.0],
        }
    )
    engine = ForecastEngine(h=6, freq="MS", n_windows=1)
    with pytest.raises(ContractError, match="exactly h=6 rows"):
        engine.run(df, X_df=bad_x)


def test_engine_rejects_unexpected_x_df(regular_series_df):
    h = 6
    future_dates = _future_months(regular_series_df["ds"].max(), h)
    uids = regular_series_df["unique_id"].unique().to_list()
    x_df = pl.DataFrame(
        {
            "unique_id": [u for u in uids for _ in range(h)],
            "ds": future_dates * len(uids),
            "extra": [0.0] * (h * len(uids)),
        }
    )
    engine = ForecastEngine(h=h, freq="1mo", n_windows=1)
    with pytest.raises(ValueError, match="no exogenous regressor columns"):
        engine.run(regular_series_df, X_df=x_df)


def test_engine_skips_foundation_model_when_regressors_present(monkeypatch):
    df = _promo_series_df()
    h = 6
    future_dates = _future_months(df["ds"].max(), h)
    x_df = pl.DataFrame(
        {"unique_id": ["sku1"] * h, "ds": future_dates, "promo": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("foundation model should be skipped when regressors are present")

    monkeypatch.setattr(selection, "evaluate_foundation_model", fail_if_called)

    engine = ForecastEngine(h=h, freq="MS", n_windows=1, use_foundation_model=True)
    result = engine.run(df, X_df=x_df)
    assert result.selection.row(0, named=True)["model"] != "T0"
