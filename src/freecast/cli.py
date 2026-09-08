"""The ``freecast`` command-line interface."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import typer

from freecast.engine import ForecastEngine

app = typer.Typer(add_completion=False, no_args_is_help=True)
override_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(override_app, name="override", help="Manage the planner-override audit trail.")


def _read(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path)
    else:
        raise typer.BadParameter(f"Unsupported input format: {path.suffix}. Use .csv or .parquet.")

    if "ds" in df.columns:
        from freecast.contract import parse_ds_column

        df = parse_ds_column(df)
    return df


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
        ...,
        "--freq",
        "-f",
        help="Frequency, pandas- or Polars-style: e.g. 'MS'/'D'/'W'/'Q' or '1mo'/'1d'/'1w'/'1q'.",
    ),
    output_dir: Path = typer.Option(Path("freecast_output"), "--output-dir", "-o"),
    season_length: int = typer.Option(
        None,
        "--season-length",
        help=(
            "Override the seasonal period (default: inferred from --freq, e.g. 12 for "
            "monthly). Set this if --freq is a label with no real periodicity behind it "
            "(e.g. synthetic daily dates over data with no weekly pattern)."
        ),
    ),
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
    use_foundation_model: bool = typer.Option(
        False,
        "--use-foundation-model",
        help=(
            "Also try The Forecasting Company's t0 zero-shot model as a candidate "
            "(requires the 'foundation' extra and Hugging Face access; only supports "
            "--levels of 80 and/or 50)."
        ),
    ),
) -> None:
    """Run the full freecast pipeline on a series file and write results to OUTPUT_DIR."""
    level_list = tuple(int(x) for x in levels.split(","))
    df = _read(input_path)

    engine = ForecastEngine(
        h=horizon,
        freq=freq,
        season_length=season_length,
        levels=level_list,
        metric=metric,
        n_windows=n_windows,
        min_history=min_history,
        on_error=on_error,
        n_jobs=n_jobs,
        use_foundation_model=use_foundation_model,
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
    output_dir: Path = typer.Option(Path("bench_results"), "--output-dir", "-o"),
) -> None:
    """Run freecast against a public M-competition dataset and report accuracy."""
    from freecast.bench.runner import run_benchmark

    result = run_benchmark(dataset=dataset, group=group, output_dir=output_dir)
    typer.echo(result)


@override_app.command("add")
def override_add(
    unique_id: str = typer.Option(..., "--unique-id"),
    ds: dt.datetime = typer.Option(..., "--ds", help="Date being overridden, e.g. 2024-01-01."),
    model: str = typer.Option(..., "--model", help="Model that produced the baseline forecast."),
    baseline: float = typer.Option(
        ..., "--baseline", help="The statistical forecast being overridden."
    ),
    value: float = typer.Option(..., "--value", help="The planner's override value."),
    author: str = typer.Option(..., "--author", help="Who is making this override."),
    reason: str = typer.Option(None, "--reason"),
    db: Path = typer.Option(Path("freecast_overrides.duckdb"), "--db"),
) -> None:
    """Record a planner override, superseding any prior active override for the same series/date."""
    from freecast.overrides import OverrideStore

    with OverrideStore(db) as store:
        result = store.add(
            unique_id=unique_id,
            ds=ds.date(),
            model=model,
            baseline_y_hat=baseline,
            override_y_hat=value,
            author=author,
            reason=reason,
        )
    typer.echo(f"Recorded override #{result.override_id} for {unique_id}@{ds.date()} by {author}.")


@override_app.command("log")
def override_log(
    unique_id: str = typer.Option(None, "--unique-id"),
    ds: dt.datetime = typer.Option(None, "--ds"),
    db: Path = typer.Option(Path("freecast_overrides.duckdb"), "--db"),
) -> None:
    """Print the full override audit trail (every version, superseded or not)."""
    from freecast.overrides import OverrideStore

    with OverrideStore(db) as store:
        log = store.audit_log(unique_id=unique_id, ds=ds.date() if ds else None)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=1000):
        typer.echo(str(log))


@app.command()
def fva(
    train_path: Path = typer.Option(..., "--train", help="History used to fit the naive baseline."),
    forecasts_path: Path = typer.Option(
        ..., "--forecasts", help="Statistical forecast to evaluate."
    ),
    actuals_path: Path = typer.Option(
        ..., "--actuals", help="Realized values for the forecast period."
    ),
    freq: str = typer.Option(..., "--freq", "-f"),
    season_length: int = typer.Option(1, "--season-length"),
    metric: str = typer.Option("mae", "--metric", help="'mae' or 'mape'."),
    overrides_db: Path = typer.Option(
        None, "--overrides-db", help="Also score the active overrides in this DuckDB file."
    ),
) -> None:
    """Report Forecast Value Added: naive vs. statistical vs. (optionally) override accuracy."""
    from freecast.fva import compute_fva
    from freecast.overrides import OverrideStore

    train_df = _read(train_path)
    forecasts_df = _read(forecasts_path)
    actuals_df = _read(actuals_path)

    overrides_df = None
    if overrides_db is not None:
        with OverrideStore(overrides_db) as store:
            overrides_df = store.active_overrides().select(["unique_id", "ds", "override_y_hat"])

    result = compute_fva(
        train_df,
        forecasts_df,
        actuals_df,
        freq=freq,
        season_length=season_length,
        overrides=overrides_df,
        metric=metric,
    )
    typer.echo(f"Overall ({metric}): {result.overall}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=1000):
        typer.echo(str(result.per_series))


if __name__ == "__main__":
    app()
