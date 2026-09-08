from __future__ import annotations

import datetime

import polars as pl
import pytest

from freecast.overrides import OverrideStore


@pytest.fixture
def store(tmp_path) -> OverrideStore:
    with OverrideStore(tmp_path / "overrides.duckdb") as s:
        yield s


def test_add_requires_author(store):
    with pytest.raises(ValueError, match="author"):
        store.add(
            unique_id="a",
            ds=datetime.date(2024, 1, 1),
            model="AutoETS",
            baseline_y_hat=10.0,
            override_y_hat=12.0,
            author="",
        )


def test_add_and_active_overrides(store):
    store.add(
        unique_id="a",
        ds=datetime.date(2024, 1, 1),
        model="AutoETS",
        baseline_y_hat=10.0,
        override_y_hat=12.0,
        author="planner1",
        reason="sales input",
    )
    active = store.active_overrides()
    assert active.height == 1
    row = active.row(0, named=True)
    assert row["unique_id"] == "a"
    assert row["override_y_hat"] == 12.0
    assert row["author"] == "planner1"
    assert row["superseded_by"] is None


def test_second_override_supersedes_first(store):
    first = store.add(
        unique_id="a",
        ds=datetime.date(2024, 1, 1),
        model="AutoETS",
        baseline_y_hat=10.0,
        override_y_hat=12.0,
        author="planner1",
    )
    second = store.add(
        unique_id="a",
        ds=datetime.date(2024, 1, 1),
        model="AutoETS",
        baseline_y_hat=10.0,
        override_y_hat=15.0,
        author="planner2",
        reason="revised sales input",
    )

    active = store.active_overrides()
    assert active.height == 1
    assert active.row(0, named=True)["override_id"] == second.override_id

    log = store.audit_log(unique_id="a", ds=datetime.date(2024, 1, 1))
    assert log.height == 2
    first_row = log.filter(pl.col("override_id") == first.override_id).row(0, named=True)
    assert first_row["superseded_by"] == second.override_id


def test_apply_merges_overrides_onto_forecasts(store):
    store.add(
        unique_id="a",
        ds=datetime.date(2024, 1, 1),
        model="AutoETS",
        baseline_y_hat=10.0,
        override_y_hat=12.0,
        author="planner1",
    )
    forecasts = pl.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "ds": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 2, 1),
                datetime.date(2024, 1, 1),
            ],
            "y_hat": [10.0, 11.0, 20.0],
        }
    )
    result = store.apply(forecasts)
    result = result.sort(["unique_id", "ds"])

    overridden = result.filter(
        (pl.col("unique_id") == "a") & (pl.col("ds") == datetime.date(2024, 1, 1))
    ).row(0, named=True)
    assert overridden["y_hat_final"] == 12.0
    assert overridden["is_overridden"] is True

    untouched = result.filter(pl.col("unique_id") == "b").row(0, named=True)
    assert untouched["y_hat_final"] == 20.0
    assert untouched["is_overridden"] is False
