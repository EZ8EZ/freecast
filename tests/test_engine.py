from __future__ import annotations

import datetime

import polars as pl

from freecast import selection
from freecast.engine import ForecastEngine, infer_season_length


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


def test_infer_season_length():
    assert infer_season_length("MS") == 12
    assert infer_season_length("1mo") == 1  # unknown key falls back to non-seasonal
    assert infer_season_length("D") == 7
    assert infer_season_length("Q") == 4
    assert infer_season_length(1) == 1


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
