from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from freecast.fva import compute_fva


def _dates(n: int, start: datetime.date = datetime.date(2020, 1, 1)) -> list[datetime.date]:
    return [start + datetime.timedelta(days=i) for i in range(n)]


@pytest.fixture
def fva_setup():
    rng = np.random.default_rng(0)
    n_train = 60
    train_dates = _dates(n_train)
    y_train = 100 + rng.normal(0, 1, n_train)
    train_df = pl.DataFrame({"unique_id": ["a"] * n_train, "ds": train_dates, "y": y_train})

    h = 7
    future_dates = _dates(h, start=train_dates[-1] + datetime.timedelta(days=1))
    y_actual = 100 + rng.normal(0, 1, h)
    actuals = pl.DataFrame({"unique_id": ["a"] * h, "ds": future_dates, "y": y_actual})

    # A "perfect" statistical forecast should beat the naive (last-value) baseline
    # whenever the series isn't a random walk, which this smooth-mean series isn't.
    forecasts = pl.DataFrame(
        {"unique_id": ["a"] * h, "ds": future_dates, "y_hat": y_actual.tolist()}
    )
    return train_df, forecasts, actuals


def test_compute_fva_perfect_forecast_beats_naive(fva_setup):
    train_df, forecasts, actuals = fva_setup
    result = compute_fva(train_df, forecasts, actuals, freq="1d", season_length=1)

    row = result.per_series.row(0, named=True)
    assert row["statistical_error"] == pytest.approx(0.0, abs=1e-9)
    assert row["statistical_fva"] > 0
    assert result.metric == "mae"


def test_compute_fva_accepts_pandas_style_freq(fva_setup):
    # The CLI's own --freq help text and README example use pandas-style
    # aliases like "D"; compute_fva must accept them directly rather than
    # only the Polars-style form StatsForecast needs internally.
    train_df, forecasts, actuals = fva_setup
    result = compute_fva(train_df, forecasts, actuals, freq="D", season_length=1)

    row = result.per_series.row(0, named=True)
    assert row["statistical_error"] == pytest.approx(0.0, abs=1e-9)
    assert "statistical_fva" in result.overall


def test_compute_fva_with_bad_override_shows_negative_fva(fva_setup):
    train_df, forecasts, actuals = fva_setup
    h = forecasts.height
    overrides = pl.DataFrame(
        {
            "unique_id": ["a"] * h,
            "ds": forecasts["ds"].to_list(),
            "override_y_hat": [0.0] * h,  # a deliberately terrible override
        }
    )
    result = compute_fva(train_df, forecasts, actuals, freq="1d", overrides=overrides)

    row = result.per_series.row(0, named=True)
    assert row["override_error"] > row["statistical_error"]
    assert row["override_fva"] < 0


def test_compute_fva_unknown_metric_raises(fva_setup):
    train_df, forecasts, actuals = fva_setup
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_fva(train_df, forecasts, actuals, freq="1d", metric="rmse")
