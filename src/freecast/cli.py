"""The ``freecast`` command-line interface."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from freecast.engine import ForecastEngine

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _read(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise typer.BadParameter(f"Unsupported input format: {path.suffix}. Use .csv or .parquet.")


def _write(df: pl.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".parquet":
        df.write_parquet(path)
    else:
        df.write_csv(path)


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="CSV or Parquet file with (unique_id, ds, y)."),
    horizon: int = typer.Option(..., "--horizon", "-h", help="Forecast horizon."),
    freq: str = typer.Option(
        ..., "--freq", "-f", help="Pandas-style frequency, e.g. 'MS', 'D', 'W'."
    ),
    output_dir: Path = typer.Option(Path("freecast_output"), "--output-dir", "-o"),
    metric: str = typer.Option(
        "mase", "--metric", help="Model-selection metric: mase|rmsse|smape|bias."
    ),
    levels: str = typer.Option(
        "80,95", "--levels", help="Comma-separated prediction-interval levels."
    ),
    n_windows: int = typer.Option(
        2, "--cv-windows", help="Rolling-origin CV windows for backtesting."
    ),
    min_history: int = typer.Option(
        6, "--min-history", help="Minimum observations required per series."
    ),
    on_error: str = typer.Option("raise", "--on-error", help="'raise' or 'drop' invalid series."),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Parallel workers; -1 uses all cores."),
) -> None:
    """Run the full freecast pipeline on a series file and write results to OUTPUT_DIR."""
    level_list = tuple(int(x) for x in levels.split(","))
    df = _read(input_path)

    engine = ForecastEngine(
        h=horizon,
        freq=freq,
        levels=level_list,
        metric=metric,
        n_windows=n_windows,
        min_history=min_history,
        on_error=on_error,
        n_jobs=n_jobs,
    )
    result = engine.run(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(result.forecasts, output_dir / "forecasts.parquet")
    _write(result.selection, output_dir / "model_selection.parquet")
    _write(result.classification, output_dir / "demand_classification.parquet")

    n_dropped = len(result.validation.dropped_series)
    typer.echo(
        f"Forecasted {result.validation.n_series} series "
        f"({n_dropped} dropped) at horizon={horizon}, freq={freq!r}."
    )
    typer.echo(f"Wrote forecasts, model_selection, demand_classification to {output_dir}/")


@app.command()
def bench(
    dataset: str = typer.Argument(..., help="Benchmark dataset: m3, m4, or tourism."),
    group: str = typer.Option(None, "--group", help="Optional sub-group, e.g. 'Monthly' for M3."),
    output_dir: Path = typer.Option(Path("bench/results"), "--output-dir", "-o"),
) -> None:
    """Run freecast against a public M-competition dataset and report accuracy."""
    from bench.runner import run_benchmark

    result = run_benchmark(dataset=dataset, group=group, output_dir=output_dir)
    typer.echo(result)


if __name__ == "__main__":
    app()
