from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest


def month_range(start: datetime.date, n: int) -> list[datetime.date]:
    dates = []
    y, m = start.year, start.month
    for _ in range(n):
        dates.append(datetime.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


@pytest.fixture
def regular_series_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 48
    dates = month_range(datetime.date(2020, 1, 1), n)
    rows = []
    for i in range(4):
        y = 100 + 10 * np.sin(np.arange(n) / 12 * 2 * np.pi) + rng.normal(0, 5, n) + i * 20
        rows.append(pl.DataFrame({"unique_id": f"series_{i}", "ds": dates, "y": y}))
    return pl.concat(rows)


@pytest.fixture
def intermittent_series_df() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    n = 48
    dates = month_range(datetime.date(2020, 1, 1), n)
    y = np.zeros(n)
    idx = rng.choice(n, 8, replace=False)
    y[idx] = rng.integers(1, 10, 8).astype(float)
    return pl.DataFrame({"unique_id": "series_sparse", "ds": dates, "y": y})


@pytest.fixture
def mixed_series_df(regular_series_df, intermittent_series_df) -> pl.DataFrame:
    return pl.concat([regular_series_df, intermittent_series_df])
