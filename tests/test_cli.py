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
