from __future__ import annotations

import datetime

import polars as pl
import pytest

from freecast.contract import ContractError, validate


def _base_df() -> pl.DataFrame:
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    return pl.DataFrame({"unique_id": ["a"] * 10, "ds": dates, "y": list(range(10))})


def test_valid_frame_passes():
    df = _base_df()
    clean, report = validate(df, min_history=3)
    assert report.n_series == 1
    assert report.n_rows == 10
    assert report.dropped_series == {}


def test_missing_required_column_raises():
    df = _base_df().drop("y")
    with pytest.raises(ContractError, match="Missing required column"):
        validate(df)


def test_empty_frame_raises():
    df = _base_df().head(0)
    with pytest.raises(ContractError, match="empty"):
        validate(df)


def test_non_numeric_y_raises():
    df = _base_df().with_columns(pl.col("y").cast(pl.Utf8))
    df = df.with_columns(pl.Series("y", ["x"] * 10))
    with pytest.raises(ContractError, match="non-numeric"):
        validate(df)


def test_null_values_raise():
    df = _base_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("y")).alias("y")
    )
    with pytest.raises(ContractError, match="Null values"):
        validate(df)


def test_duplicate_timestamps_raise():
    df = _base_df()
    dup_row = df.head(1)
    df = pl.concat([df, dup_row])
    with pytest.raises(ContractError, match="Duplicate"):
        validate(df)


def test_duplicate_timestamps_dropped_on_error_drop():
    a = _base_df()
    dup_row = a.head(1)
    a = pl.concat([a, dup_row])
    b = _base_df().with_columns(pl.lit("b").alias("unique_id"))
    df = pl.concat([a, b])
    clean, report = validate(df, min_history=3, on_error="drop")
    assert report.n_series == 1
    assert "a" in report.dropped_series
    assert set(clean["unique_id"].unique().to_list()) == {"b"}


def test_missing_dates_gap_raises():
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    del dates[5]
    y = list(range(9))
    df = pl.DataFrame({"unique_id": ["a"] * 9, "ds": dates, "y": y})
    with pytest.raises(ContractError, match="Missing dates"):
        validate(df, min_history=3)


def test_insufficient_history_raises():
    df = _base_df().head(2)
    with pytest.raises(ContractError, match="min_history"):
        validate(df, min_history=6)


def test_insufficient_history_dropped_on_error_drop():
    short = _base_df().head(2)
    long_df = _base_df().with_columns(pl.lit("b").alias("unique_id"))
    df = pl.concat([short, long_df])
    clean, report = validate(df, min_history=6, on_error="drop")
    assert report.n_series == 1
    assert "a" in report.dropped_series


def test_multi_series_ok():
    a = _base_df()
    b = _base_df().with_columns(pl.lit("b").alias("unique_id"))
    df = pl.concat([a, b])
    clean, report = validate(df, min_history=3)
    assert report.n_series == 2
