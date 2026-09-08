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
