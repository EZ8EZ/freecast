from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from freecast.cli import app

runner = CliRunner()


def test_cli_run_end_to_end(tmp_path: Path, mixed_series_df):
    input_path = tmp_path / "series.csv"
    mixed_series_df.write_csv(input_path)
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(input_path),
            "--horizon",
            "6",
            "--freq",
            "1mo",
            "--output-dir",
            str(output_dir),
            "--cv-windows",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "forecasts.parquet").exists()
    assert (output_dir / "model_selection.parquet").exists()
    assert (output_dir / "demand_classification.parquet").exists()

    import polars as pl

    forecasts = pl.read_parquet(output_dir / "forecasts.parquet")
    assert forecasts.height > 0


def test_cli_unsupported_format(tmp_path: Path):
    bad_file = tmp_path / "series.txt"
    bad_file.write_text("unique_id,ds,y\n")
    result = runner.invoke(app, ["run", str(bad_file), "--horizon", "3", "--freq", "1mo"])
    assert result.exit_code != 0


def test_cli_override_add_and_log(tmp_path: Path):
    db = tmp_path / "overrides.duckdb"
    add_result = runner.invoke(
        app,
        [
            "override",
            "add",
            "--unique-id",
            "a",
            "--ds",
            "2024-01-01",
            "--model",
            "AutoETS",
            "--baseline",
            "10.0",
            "--value",
            "12.0",
            "--author",
            "eric",
            "--reason",
            "sales input",
            "--db",
            str(db),
        ],
    )
    assert add_result.exit_code == 0, add_result.output
    assert "Recorded override #1" in add_result.output

    log_result = runner.invoke(app, ["override", "log", "--db", str(db)])
    assert log_result.exit_code == 0, log_result.output
    assert "eric" in log_result.output
    assert "sales input" in log_result.output


def test_cli_fva(tmp_path: Path):
    import datetime

    import numpy as np
    import polars as pl

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

    result = runner.invoke(
        app,
        [
            "fva",
            "--train",
            str(train_path),
            "--forecasts",
            str(forecasts_path),
            "--actuals",
            str(actuals_path),
            "--freq",
            "1d",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "statistical_fva" in result.output


def test_cli_reconcile(tmp_path: Path):
    import datetime

    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    n = 48
    dates = []
    y, m = 2020, 1
    for _ in range(n):
        dates.append(datetime.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1

    rows = []
    for cat in ["A", "B"]:
        for region in ["X", "Y"]:
            base = 100 + (20 if cat == "B" else 0) + (10 if region == "Y" else 0)
            for i, d in enumerate(dates):
                val = base + 10 * np.sin(i / 12 * 2 * np.pi) + rng.normal(0, 3)
                rows.append({"category": cat, "region": region, "ds": d, "y": val})
    input_path = tmp_path / "hierarchy.csv"
    pl.DataFrame(rows).write_csv(input_path)
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "reconcile",
            str(input_path),
            "--hierarchy",
            "category,region",
            "--horizon",
            "6",
            "--freq",
            "1mo",
            "--output-dir",
            str(output_dir),
            "--cv-windows",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Reconciled 6 series across 2 hierarchy levels" in result.output
    assert (output_dir / "reconciled_forecasts.parquet").exists()

    forecasts = pl.read_parquet(output_dir / "reconciled_forecasts.parquet")
    assert "y_hat/BottomUp" in forecasts.columns
    assert set(forecasts["unique_id"].unique().to_list()) == {"A", "B", "A/X", "A/Y", "B/X", "B/Y"}


def test_cli_run_with_regressors(tmp_path: Path):
    import datetime

    import numpy as np
    import polars as pl

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

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            str(input_path),
            "--horizon",
            str(h),
            "--freq",
            "MS",
            "--output-dir",
            str(output_dir),
            "--cv-windows",
            "1",
            "--regressors-path",
            str(regressors_path),
        ],
    )
    assert result.exit_code == 0, result.output
    forecasts = pl.read_parquet(output_dir / "forecasts.parquet")
    assert forecasts.height == h


def test_cli_run_missing_regressors_fails(tmp_path: Path):
    import polars as pl

    input_path = tmp_path / "series.csv"
    pl.DataFrame(
        {
            "unique_id": ["a"] * 10,
            "ds": [f"2024-{i + 1:02d}-01" for i in range(10)],
            "y": list(range(10)),
            "promo": [0.0] * 10,
        }
    ).write_csv(input_path)

    result = runner.invoke(app, ["run", str(input_path), "--horizon", "3", "--freq", "MS"])
    assert result.exit_code != 0
