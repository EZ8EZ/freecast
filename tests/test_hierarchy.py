from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from freecast.hierarchy import (
    BOTTOM_UP_RECONCILER_NAME,
    DEFAULT_RECONCILER_NAME,
    build_hierarchy,
    check_coherence,
    reconcile,
)


def _month_range(n: int) -> list[datetime.date]:
    dates = []
    y, m = 2020, 1
    for _ in range(n):
        dates.append(datetime.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


@pytest.fixture
def hierarchy_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 48
    dates = _month_range(n)
    rows = []
    for cat in ["A", "B"]:
        for region in ["X", "Y"]:
            base = 100 + (20 if cat == "B" else 0) + (10 if region == "Y" else 0)
            for i, d in enumerate(dates):
                val = base + 10 * np.sin(i / 12 * 2 * np.pi) + rng.normal(0, 3)
                rows.append({"category": cat, "region": region, "ds": d, "y": val})
    return pl.DataFrame(rows)


def test_build_hierarchy_aggregates_every_level(hierarchy_df):
    aggregated, s_df, tags = build_hierarchy(
        hierarchy_df, spec=[["category"], ["category", "region"]]
    )

    assert set(tags.keys()) == {"category", "category/region"}
    assert set(tags["category"]) == {"A", "B"}
    assert set(tags["category/region"]) == {"A/X", "A/Y", "B/X", "B/Y"}
    assert set(aggregated["unique_id"].unique().to_list()) == {"A", "B", "A/X", "A/Y", "B/X", "B/Y"}
    assert s_df.height == 6  # 2 top + 4 bottom


def test_build_hierarchy_requires_spec(hierarchy_df):
    with pytest.raises(ValueError, match="spec must list"):
        build_hierarchy(hierarchy_df, spec=[])


def test_reconcile_bottom_up_is_coherent(hierarchy_df):
    result = reconcile(
        hierarchy_df, spec=[["category"], ["category", "region"]], h=6, freq="1mo", n_windows=1
    )
    assert check_coherence(result.forecasts, result.s_matrix, BOTTOM_UP_RECONCILER_NAME)


def test_reconcile_mintrace_is_coherent(hierarchy_df):
    result = reconcile(
        hierarchy_df, spec=[["category"], ["category", "region"]], h=6, freq="1mo", n_windows=1
    )
    assert check_coherence(result.forecasts, result.s_matrix, DEFAULT_RECONCILER_NAME)


def test_unreconciled_forecast_is_not_generally_coherent(hierarchy_df):
    # Each level is forecast independently by the engine, so summing the
    # unreconciled y_hat up from the bottom generally will NOT match the
    # independently-forecast top level -- that's the incoherence problem
    # reconciliation exists to fix.
    result = reconcile(
        hierarchy_df, spec=[["category"], ["category", "region"]], h=6, freq="1mo", n_windows=1
    )
    assert not check_coherence(result.forecasts, result.s_matrix, "y_hat")


def test_reconcile_exposes_engine_result(hierarchy_df):
    result = reconcile(
        hierarchy_df, spec=[["category"], ["category", "region"]], h=6, freq="1mo", n_windows=1
    )
    all_ids = {"A", "B", "A/X", "A/Y", "B/X", "B/Y"}
    assert set(result.engine_result.selection["unique_id"].to_list()) == all_ids
    assert set(result.aggregated["unique_id"].unique().to_list()) == all_ids
