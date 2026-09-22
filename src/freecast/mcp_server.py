"""MCP server exposing freecast as tools for any MCP-aware client.

This is a thin layer over the existing library — every tool here is a
direct wrapper around ``ForecastEngine``, ``OverrideStore``, ``compute_fva``,
or ``freecast.hierarchy.reconcile``. No forecasting, override, or
reconciliation logic lives in this file; it only handles MCP plumbing
(input validation, reading/writing local files, JSON-shaping results).

Per the locked architectural decision, this is *not* a hosted chatbot or a
bundled UI: it's a local process (stdio transport) that any MCP client
(Claude Code, Claude Desktop, etc.) attaches to, pointed at a local project
directory of CSV/Parquet files — the same headless, local-first model the
rest of freecast uses.

Run with: ``freecast mcp`` (see cli.py) or ``python -m freecast.mcp_server``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Literal

import polars as pl
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

mcp = FastMCP("freecast_mcp")

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _read_df(path: str) -> pl.DataFrame:
    from freecast.contract import parse_ds_column

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pl.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pl.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format {p.suffix!r}; use .csv or .parquet.")
    if "ds" in df.columns:
        df = parse_ds_column(df)
    return df


def _write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _to_json(df: pl.DataFrame, *, limit: int) -> dict:
    """Cap rows returned inline; the full result is always on disk too."""
    truncated = df.height > limit
    return {
        "row_count": df.height,
        "truncated": truncated,
        "rows": df.head(limit).to_dicts(),
    }


class RunForecastInput(BaseModel):
    """Input for running the core forecasting pipeline."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ..., description="Path to a CSV or Parquet file with columns (unique_id, ds, y)."
    )
    horizon: int = Field(..., ge=1, description="Forecast horizon, in periods.")
    freq: str = Field(
        ..., description="Series frequency, pandas- or Polars-style, e.g. 'MS', 'D', '1mo'."
    )
    output_dir: str = Field(
        default="freecast_output",
        description="Directory to write forecasts/model_selection/classification files to.",
    )
    metric: Literal["mase", "rmsse", "smape", "bias"] = Field(
        default="mase", description="Model-selection metric; lower is better for all of them."
    )
    levels: list[int] = Field(
        default_factory=lambda: [80, 95], description="Prediction-interval levels to produce."
    )
    min_history: int = Field(
        default=6, ge=2, description="Minimum observations required per series."
    )
    on_error: Literal["raise", "drop"] = Field(
        default="raise", description="'raise' aborts on invalid series; 'drop' excludes them."
    )
    regressors_path: str | None = Field(
        default=None,
        description="CSV/Parquet with known future values (unique_id, ds, plus every extra "
        "column in input_path) — required if input_path has exogenous regressor columns "
        "beyond unique_id/ds/y, e.g. a promo calendar.",
    )


@mcp.tool(
    name="freecast_run_forecast",
    annotations=WRITE,
    description=(
        "Run freecast's full forecasting pipeline (validate, classify demand type, "
        "cross-validate and select a model per series, forecast with conformal "
        "prediction intervals) on a batch of series and write the results to disk."
    ),
)
def run_forecast(params: RunForecastInput) -> str:
    """Run the core engine end to end and write forecasts/selection/classification to disk.

    Returns a JSON string: {n_series, n_dropped, output_dir, files: {...}}.
    Use freecast_get_forecast afterwards to read specific series back out
    without re-running the pipeline.
    """
    from freecast.engine import ForecastEngine

    df = _read_df(params.input_path)
    X_df = _read_df(params.regressors_path) if params.regressors_path is not None else None
    engine = ForecastEngine(
        h=params.horizon,
        freq=params.freq,
        levels=tuple(params.levels),
        metric=params.metric,
        min_history=params.min_history,
        on_error=params.on_error,
    )
    result = engine.run(df, X_df=X_df)

    out = Path(params.output_dir)
    _write_df(result.forecasts, out / "forecasts.parquet")
    _write_df(result.selection, out / "model_selection.parquet")
    _write_df(result.classification, out / "demand_classification.parquet")

    return json.dumps(
        {
            "n_series": result.validation.n_series,
            "n_dropped": len(result.validation.dropped_series),
            "dropped_series": result.validation.dropped_series,
            "output_dir": str(out),
            "files": {
                "forecasts": str(out / "forecasts.parquet"),
                "model_selection": str(out / "model_selection.parquet"),
                "demand_classification": str(out / "demand_classification.parquet"),
            },
        },
        indent=2,
    )


class GetForecastInput(BaseModel):
    """Input for reading back forecasts written by freecast_run_forecast."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        ..., description="The output_dir a prior freecast_run_forecast call wrote to."
    )
    unique_ids: list[str] | None = Field(
        default=None,
        description="Restrict to these series' unique_ids; omit to return all (subject to limit).",
    )
    limit: int = Field(
        default=500, ge=1, le=5000, description="Maximum rows to return inline in the response."
    )


@mcp.tool(
    name="freecast_get_forecast",
    annotations=READ_ONLY,
    description="Read back forecasts (and the model chosen per series) from a prior run.",
)
def get_forecast(params: GetForecastInput) -> str:
    """Query forecasts.parquet / model_selection.parquet from a completed run.

    Returns a JSON string: {forecasts: {row_count, truncated, rows}, selection: {...}}.
    ``rows`` are capped at ``limit`` — the parquet files on disk always hold
    the full result, so re-query with a narrower unique_ids filter for series
    that were truncated.
    """
    out = Path(params.output_dir)
    forecasts = pl.read_parquet(out / "forecasts.parquet")
    selection = pl.read_parquet(out / "model_selection.parquet")

    if params.unique_ids is not None:
        forecasts = forecasts.filter(pl.col("unique_id").is_in(params.unique_ids))
        selection = selection.filter(pl.col("unique_id").is_in(params.unique_ids))

    return json.dumps(
        {
            "forecasts": _to_json(forecasts, limit=params.limit),
            "selection": _to_json(selection, limit=params.limit),
        },
        indent=2,
        default=str,
    )


class AddOverrideInput(BaseModel):
    """Input for recording a planner override."""

    model_config = ConfigDict(extra="forbid")

    unique_id: str = Field(..., description="The series being overridden.")
    ds: str = Field(
        ..., description="The date being overridden, as an ISO date string (YYYY-MM-DD)."
    )
    model: str = Field(
        ..., description="The model that produced the baseline forecast, e.g. 'AutoETS'."
    )
    baseline_y_hat: float = Field(
        ..., description="The statistical forecast value being overridden."
    )
    override_y_hat: float = Field(..., description="The planner's replacement value.")
    author: str = Field(
        ..., min_length=1, description="Who is making this override — required for attribution."
    )
    reason: str | None = Field(default=None, description="Why the override is being made.")
    db_path: str = Field(
        default="freecast_overrides.duckdb",
        description="Path to the override audit-trail database.",
    )


@mcp.tool(
    name="freecast_add_override",
    annotations=WRITE,
    description="Record a planner override, with a durable, attributable audit trail.",
)
def add_override(params: AddOverrideInput) -> str:
    """Append a new override, superseding any prior active override for the same series/date.

    Returns a JSON string: {override_id, unique_id, ds, author, superseded: bool}.
    """
    from freecast.overrides import OverrideStore

    try:
        ds = dt.date.fromisoformat(params.ds)
    except ValueError as exc:
        raise ValueError(f"ds must be an ISO date (YYYY-MM-DD); got {params.ds!r}") from exc

    with OverrideStore(params.db_path) as store:
        prior = store.audit_log(unique_id=params.unique_id, ds=ds)
        result = store.add(
            unique_id=params.unique_id,
            ds=ds,
            model=params.model,
            baseline_y_hat=params.baseline_y_hat,
            override_y_hat=params.override_y_hat,
            author=params.author,
            reason=params.reason,
        )

    return json.dumps(
        {
            "override_id": result.override_id,
            "unique_id": result.unique_id,
            "ds": result.ds.isoformat(),
            "override_y_hat": result.override_y_hat,
            "author": result.author,
            "superseded_prior": prior.height > 0,
        },
        indent=2,
    )


class ListOverridesInput(BaseModel):
    """Input for listing overrides from the audit trail."""

    model_config = ConfigDict(extra="forbid")

    unique_id: str | None = Field(default=None, description="Restrict to this series.")
    active_only: bool = Field(
        default=True,
        description="If true, return only the current override per series/date. "
        "If false, return the full history including superseded overrides.",
    )
    db_path: str = Field(
        default="freecast_overrides.duckdb",
        description="Path to the override audit-trail database.",
    )


@mcp.tool(
    name="freecast_list_overrides",
    annotations=READ_ONLY,
    description="List planner overrides — either just the active ones, or the full audit history.",
)
def list_overrides(params: ListOverridesInput) -> str:
    """Read the override audit trail.

    Returns a JSON string: {row_count, truncated, rows} (see freecast_add_override
    for the shape of each row — unique_id, ds, model, baseline_y_hat,
    override_y_hat, author, reason, created_at, superseded_by).
    """
    from freecast.overrides import OverrideStore

    with OverrideStore(params.db_path) as store:
        log = (
            store.active_overrides()
            if params.active_only
            else store.audit_log(unique_id=params.unique_id)
        )
    if params.active_only and params.unique_id is not None:
        log = log.filter(pl.col("unique_id") == params.unique_id)

    return json.dumps(_to_json(log, limit=1000), indent=2, default=str)


class GetFvaReportInput(BaseModel):
    """Input for computing Forecast Value Added."""

    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(
        ..., description="History used to fit the naive baseline (unique_id, ds, y)."
    )
    forecasts_path: str = Field(
        ..., description="Statistical forecast being evaluated (unique_id, ds, y_hat)."
    )
    actuals_path: str = Field(
        ..., description="Realized values for the forecast period (unique_id, ds, y)."
    )
    freq: str = Field(..., description="Series frequency, pandas- or Polars-style.")
    season_length: int = Field(
        default=1, ge=1, description="Seasonal period for the naive baseline."
    )
    metric: Literal["mae", "mape"] = Field(default="mae")
    overrides_db_path: str | None = Field(
        default=None, description="Also score active overrides from this audit-trail database."
    )


@mcp.tool(
    name="freecast_get_fva_report",
    annotations=READ_ONLY,
    description=(
        "Report Forecast Value Added: does the statistical model, or a planner's override, "
        "actually beat a naive baseline? Positive means yes; negative means that layer is "
        "making forecasts worse."
    ),
)
def get_fva_report(params: GetFvaReportInput) -> str:
    """Compute per-series and overall FVA.

    Returns a JSON string: {metric, overall: {...}, per_series: {row_count, truncated, rows}}.
    """
    from freecast.fva import compute_fva
    from freecast.overrides import OverrideStore

    train_df = _read_df(params.train_path)
    forecasts_df = _read_df(params.forecasts_path)
    actuals_df = _read_df(params.actuals_path)

    overrides_df = None
    if params.overrides_db_path is not None:
        with OverrideStore(params.overrides_db_path) as store:
            overrides_df = store.active_overrides().select(["unique_id", "ds", "override_y_hat"])

    result = compute_fva(
        train_df,
        forecasts_df,
        actuals_df,
        freq=params.freq,
        season_length=params.season_length,
        overrides=overrides_df,
        metric=params.metric,
    )

    return json.dumps(
        {
            "metric": result.metric,
            "overall": result.overall,
            "per_series": _to_json(result.per_series, limit=1000),
        },
        indent=2,
        default=str,
    )


class ReconcileHierarchyInput(BaseModel):
    """Input for hierarchical reconciliation."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ..., description="Bottom-level series with grouping columns, ds, y (CSV or Parquet)."
    )
    hierarchy_columns: list[str] = Field(
        ...,
        min_length=1,
        description="Grouping columns, top level to bottom, e.g. ['category', 'region']. "
        "Every prefix becomes a hierarchy level; the full list is the bottom level.",
    )
    horizon: int = Field(..., ge=1)
    freq: str = Field(..., description="Series frequency, pandas- or Polars-style.")
    output_dir: str = Field(default="freecast_output", description="Directory to write results to.")


@mcp.tool(
    name="freecast_reconcile_hierarchy",
    annotations=WRITE,
    description=(
        "Forecast and reconcile a product/region/etc. hierarchy so bottom-level forecasts "
        "sum coherently into every parent level's total — required for S&OP and finance review."
    ),
)
def reconcile_hierarchy(params: ReconcileHierarchyInput) -> str:
    """Aggregate, forecast every level, and reconcile with MinTrace (structural scaling).

    Returns a JSON string: {n_series, levels, coherent, output_file}.
    """
    from freecast.hierarchy import DEFAULT_RECONCILER_NAME, check_coherence
    from freecast.hierarchy import reconcile as reconcile_hierarchy_fn

    df = _read_df(params.input_path)
    columns = params.hierarchy_columns
    spec = [columns[: i + 1] for i in range(len(columns))]

    result = reconcile_hierarchy_fn(df, spec, h=params.horizon, freq=params.freq)

    out_path = Path(params.output_dir) / "reconciled_forecasts.parquet"
    _write_df(result.forecasts, out_path)

    coherent = check_coherence(result.forecasts, result.s_matrix, DEFAULT_RECONCILER_NAME)

    return json.dumps(
        {
            "n_series": result.aggregated["unique_id"].n_unique(),
            "levels": list(result.tags.keys()),
            "reconciler": DEFAULT_RECONCILER_NAME,
            "coherent": coherent,
            "output_file": str(out_path),
        },
        indent=2,
    )


class GetExceptionsInput(BaseModel):
    """Input for flagging series that need a planner's review."""

    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(..., description="History the forecast was built from.")
    output_dir: str = Field(..., description="The output_dir a prior freecast_run_forecast wrote.")
    metric: Literal["mase", "rmsse"] = Field(
        default="mase", description="Model-selection metric to flag on."
    )
    accuracy_threshold: float = Field(default=1.0, description="Above this metric value, flag.")
    level: int = Field(default=80, description="Prediction-interval level for the width flag.")
    interval_width_threshold: float = Field(
        default=0.5, description="Flag when (hi-lo)/y_hat exceeds this fraction."
    )
    jump_threshold: float = Field(
        default=0.5,
        description="Flag when the forecast mean differs from the last actual "
        "by more than this fraction.",
    )
    prior_classification_path: str | None = Field(
        default=None,
        description="A demand_classification.parquet from an earlier run, to flag category "
        "changes against the current run's demand_classification.parquet in output_dir.",
    )
    limit: int = Field(default=500, ge=1, le=5000, description="Max rows returned inline.")


@mcp.tool(
    name="freecast_get_exceptions",
    annotations=READ_ONLY,
    description=(
        "Flag series worth a planner's review at scale: poor CV accuracy, wide prediction "
        "intervals, a big jump vs. the last actual, or (optionally) a changed demand-type "
        "classification. Every flag is independently interpretable, not a black-box score."
    ),
)
def get_exceptions(params: GetExceptionsInput) -> str:
    """Run freecast.exceptions.find_exceptions over a completed run's outputs.

    Returns a JSON string: {summary: {flag: count}, flagged: {row_count, truncated, rows}}.
    """
    from freecast.exceptions import find_exceptions

    out = Path(params.output_dir)
    train_df = _read_df(params.train_path)
    forecasts_df = pl.read_parquet(out / "forecasts.parquet")
    selection_df = pl.read_parquet(out / "model_selection.parquet")

    prior_classification = None
    classification = None
    if params.prior_classification_path is not None:
        prior_classification = _read_df(params.prior_classification_path)
        classification = pl.read_parquet(out / "demand_classification.parquet")

    report = find_exceptions(
        train_df,
        forecasts_df,
        selection_df,
        metric=params.metric,
        accuracy_threshold=params.accuracy_threshold,
        level=params.level,
        interval_width_threshold=params.interval_width_threshold,
        jump_threshold=params.jump_threshold,
        prior_classification=prior_classification,
        classification=classification,
    )

    return json.dumps(
        {
            "summary": report.summary,
            "flagged": _to_json(report.flagged, limit=params.limit),
        },
        indent=2,
        default=str,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
