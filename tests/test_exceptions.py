from __future__ import annotations

import datetime as dt

import polars as pl

from freecast.exceptions import find_exceptions


def _train_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "ds": [dt.date(2024, 1, i + 1) for i in range(5)] * 3,
            "y": [100.0] * 5 + [50.0] * 5 + [10.0] * 5,
        }
    )


def _forecasts_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "ds": [dt.date(2024, 2, i + 1) for i in range(3)] * 3,
            # a: accurate, tight interval, no jump.
            # b: poor accuracy, wide interval, big jump.
            # c: fine on everything.
            "y_hat": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 10.0, 10.0, 10.0],
            "lo-80": [90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 9.0, 9.0, 9.0],
            "hi-80": [110.0, 110.0, 110.0, 300.0, 300.0, 300.0, 11.0, 11.0, 11.0],
        }
    )


def _selection_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["a", "b", "c"],
            "model": ["AutoETS", "AutoETS", "AutoETS"],
            "mase": [0.8, 1.5, 0.9],
        }
    )


def test_flags_poor_accuracy() -> None:
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        _selection_df(),
        interval_width_threshold=99,
        jump_threshold=99,
    )
    assert report.summary == {"poor_accuracy": 1}
    row = report.flagged.row(0, named=True)
    assert row["unique_id"] == "b"
    assert row["flags"] == ["poor_accuracy"]


def test_flags_wide_interval() -> None:
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        _selection_df(),
        accuracy_threshold=99,
        jump_threshold=99,
    )
    assert report.summary == {"wide_interval": 1}
    row = report.flagged.row(0, named=True)
    assert row["unique_id"] == "b"


def test_flags_large_jump() -> None:
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        _selection_df(),
        accuracy_threshold=99,
        interval_width_threshold=99,
    )
    assert report.summary == {"large_jump": 1}
    row = report.flagged.row(0, named=True)
    assert row["unique_id"] == "b"


def test_combined_flags_ranked_by_count() -> None:
    report = find_exceptions(_train_df(), _forecasts_df(), _selection_df())
    assert report.flagged.height == 1
    row = report.flagged.row(0, named=True)
    assert row["unique_id"] == "b"
    assert row["n_flags"] == 3
    assert set(row["flags"]) == {"poor_accuracy", "wide_interval", "large_jump"}
    assert report.summary == {"poor_accuracy": 1, "wide_interval": 1, "large_jump": 1}


def test_no_exceptions_when_thresholds_not_tripped() -> None:
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        _selection_df(),
        accuracy_threshold=99,
        interval_width_threshold=99,
        jump_threshold=99,
    )
    assert report.flagged.height == 0
    assert report.summary == {}
    assert report.flagged.columns == ["unique_id", "flags", "n_flags"]


def test_demand_type_changed_flag() -> None:
    prior = pl.DataFrame({"unique_id": ["a", "b", "c"], "category": ["smooth"] * 3})
    current = pl.DataFrame(
        {"unique_id": ["a", "b", "c"], "category": ["smooth", "smooth", "erratic"]}
    )
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        _selection_df(),
        accuracy_threshold=99,
        interval_width_threshold=99,
        jump_threshold=99,
        prior_classification=prior,
        classification=current,
    )
    assert report.summary == {"demand_type_changed": 1}
    row = report.flagged.row(0, named=True)
    assert row["unique_id"] == "c"
    assert row["prior_category"] == "smooth"
    assert row["new_category"] == "erratic"


def test_ignores_non_scaled_metric() -> None:
    selection = pl.DataFrame(
        {"unique_id": ["a", "b", "c"], "model": ["AutoETS"] * 3, "mape": [50.0, 200.0, 5.0]}
    )
    report = find_exceptions(
        _train_df(),
        _forecasts_df(),
        selection,
        metric="mape",
        interval_width_threshold=99,
        jump_threshold=99,
    )
    assert report.summary == {}
