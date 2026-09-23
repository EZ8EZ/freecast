from __future__ import annotations

import asyncio
import datetime
import json

import numpy as np
import polars as pl
import pytest

from freecast.mcp_server import mcp


def _call(name: str, params: dict) -> dict:
    """Call an MCP tool in-process and parse its JSON text response."""
    _content, structured = asyncio.run(mcp.call_tool(name, {"params": params}))
    return json.loads(structured["result"])


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
def series_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 48
    dates = _month_range(n)
    vals = 100 + 10 * np.sin(np.arange(n) / 12 * 2 * np.pi) + rng.normal(0, 3, n)
    path = tmp_path / "series.csv"
    pl.DataFrame({"unique_id": ["a"] * n, "ds": dates, "y": vals}).write_csv(path)
    return path


def test_run_and_get_forecast(tmp_path, series_csv):
    out_dir = tmp_path / "out"
    run_result = _call(
        "freecast_run_forecast",
        {
            "input_path": str(series_csv),
            "horizon": 6,
            "freq": "MS",
            "output_dir": str(out_dir),
            "min_history": 6,
        },
    )
    assert run_result["n_series"] == 1
    assert run_result["n_dropped"] == 0

    get_result = _call("freecast_get_forecast", {"output_dir": str(out_dir)})
    assert get_result["forecasts"]["row_count"] == 6
    assert not get_result["forecasts"]["truncated"]
    assert get_result["selection"]["row_count"] == 1

    filtered = _call(
        "freecast_get_forecast", {"output_dir": str(out_dir), "unique_ids": ["nonexistent"]}
    )
    assert filtered["forecasts"]["row_count"] == 0


def test_get_forecast_respects_limit(tmp_path, series_csv):
    out_dir = tmp_path / "out"
    _call(
        "freecast_run_forecast",
        {
            "input_path": str(series_csv),
            "horizon": 6,
            "freq": "MS",
            "output_dir": str(out_dir),
        },
    )
    result = _call("freecast_get_forecast", {"output_dir": str(out_dir), "limit": 2})
    assert result["forecasts"]["row_count"] == 6
    assert result["forecasts"]["truncated"] is True
    assert len(result["forecasts"]["rows"]) == 2


def test_add_and_list_overrides(tmp_path):
    db_path = tmp_path / "overrides.duckdb"
    add_result = _call(
        "freecast_add_override",
        {
            "unique_id": "a",
            "ds": "2024-01-01",
            "model": "AutoETS",
            "baseline_y_hat": 100.0,
            "override_y_hat": 120.0,
            "author": "planner1",
            "reason": "promo",
            "db_path": str(db_path),
        },
    )
    assert add_result["override_id"] == 1
    assert add_result["superseded_prior"] is False

    second = _call(
        "freecast_add_override",
        {
            "unique_id": "a",
            "ds": "2024-01-01",
            "model": "AutoETS",
            "baseline_y_hat": 100.0,
            "override_y_hat": 130.0,
            "author": "planner2",
            "db_path": str(db_path),
        },
    )
    assert second["superseded_prior"] is True

    active = _call("freecast_list_overrides", {"db_path": str(db_path)})
    assert active["row_count"] == 1
    assert active["rows"][0]["override_y_hat"] == 130.0

    full_log = _call("freecast_list_overrides", {"db_path": str(db_path), "active_only": False})
    assert full_log["row_count"] == 2


def test_add_override_rejects_bad_date(tmp_path):
    with pytest.raises(Exception, match="ISO date"):
        _call(
            "freecast_add_override",
            {
                "unique_id": "a",
                "ds": "not-a-date",
                "model": "AutoETS",
                "baseline_y_hat": 1.0,
                "override_y_hat": 2.0,
                "author": "planner1",
                "db_path": str(tmp_path / "overrides.duckdb"),
            },
        )


def test_get_fva_report(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    train_df = pl.DataFrame({"unique_id": ["a"] * n, "ds": dates, "y": 100 + rng.normal(0, 1, n)})
    train_path = tmp_path / "train.csv"
    train_df.write_csv(train_path)

    h = 7
    future_dates = [dates[-1] + datetime.timedelta(days=i + 1) for i in range(h)]
    y_actual = 100 + rng.normal(0, 1, h)
    actuals_path = tmp_path / "actuals.csv"
    pl.DataFrame({"unique_id": ["a"] * h, "ds": future_dates, "y": y_actual}).write_csv(
        actuals_path
    )
    forecasts_path = tmp_path / "forecasts.csv"
    pl.DataFrame(
        {"unique_id": ["a"] * h, "ds": future_dates, "y_hat": y_actual.tolist()}
    ).write_csv(forecasts_path)

    result = _call(
        "freecast_get_fva_report",
        {
            "train_path": str(train_path),
            "forecasts_path": str(forecasts_path),
            "actuals_path": str(actuals_path),
            "freq": "D",
        },
    )
    assert result["metric"] == "mae"
    assert "statistical_fva" in result["overall"]
    assert result["per_series"]["row_count"] == 1


def test_reconcile_hierarchy(tmp_path):
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
    input_path = tmp_path / "hierarchy.csv"
    pl.DataFrame(rows).write_csv(input_path)
    out_dir = tmp_path / "out"

    result = _call(
        "freecast_reconcile_hierarchy",
        {
            "input_path": str(input_path),
            "hierarchy_columns": ["category", "region"],
            "horizon": 6,
            "freq": "MS",
            "output_dir": str(out_dir),
        },
    )
    assert result["n_series"] == 6
    assert set(result["levels"]) == {"category", "category/region"}
    assert result["coherent"] is True

    forecasts = pl.read_parquet(result["output_file"])
    assert set(forecasts["unique_id"].unique().to_list()) == {"A", "B", "A/X", "A/Y", "B/X", "B/Y"}


def test_run_forecast_with_regressors(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
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
    input_path = tmp_path / "series.csv"
    pl.DataFrame({"unique_id": ["sku1"] * n, "ds": dates, "y": y_vals, "promo": promo}).write_csv(
        input_path
    )

    h = 6
    future_dates = []
    yy, mm = y, m
    for _ in range(h):
        future_dates.append(datetime.date(yy, mm, 1))
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
    regressors_path = tmp_path / "regressors.csv"
    pl.DataFrame(
        {"unique_id": ["sku1"] * h, "ds": future_dates, "promo": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
    ).write_csv(regressors_path)

    out_dir = tmp_path / "out"
    result = _call(
        "freecast_run_forecast",
        {
            "input_path": str(input_path),
            "horizon": h,
            "freq": "MS",
            "output_dir": str(out_dir),
            "regressors_path": str(regressors_path),
        },
    )
    assert result["n_series"] == 1

    selection = pl.read_parquet(out_dir / "model_selection.parquet")
    assert selection.row(0, named=True)["model"] == "AutoARIMA"


def test_get_exceptions(tmp_path, series_csv):
    out_dir = tmp_path / "out"
    _call(
        "freecast_run_forecast",
        {
            "input_path": str(series_csv),
            "horizon": 6,
            "freq": "MS",
            "output_dir": str(out_dir),
            "min_history": 6,
        },
    )

    result = _call(
        "freecast_get_exceptions",
        {
            "train_path": str(series_csv),
            "output_dir": str(out_dir),
            "accuracy_threshold": -1.0,
        },
    )
    assert "poor_accuracy" in result["summary"]
    assert result["flagged"]["row_count"] >= 1


def test_run_forecast_missing_regressors_errors(tmp_path):
    input_path = tmp_path / "series.csv"
    pl.DataFrame(
        {
            "unique_id": ["a"] * 10,
            "ds": [f"2024-{i + 1:02d}-01" for i in range(10)],
            "y": list(range(10)),
            "promo": [0.0] * 10,
        }
    ).write_csv(input_path)

    with pytest.raises(Exception, match="exogenous regressor"):
        _call(
            "freecast_run_forecast",
            {"input_path": str(input_path), "horizon": 3, "freq": "MS"},
        )
