"""Data contract validation for long-format time-series input.

Expected shape: columns ``unique_id``, ``ds``, ``y``, plus any number of
optional exogenous regressor columns. Validation fails loudly and
specifically rather than silently coercing bad data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

REQUIRED_COLUMNS = ("unique_id", "ds", "y")
DEFAULT_MIN_HISTORY = 6


class ContractError(ValueError):
    """Raised when input data violates the freecast data contract."""


@dataclass
class ValidationReport:
    """Summary of a contract validation pass."""

    n_series: int
    n_rows: int
    dropped_series: dict[str, str] = field(default_factory=dict)
    """Maps unique_id -> reason, for series excluded via ``on_error='drop'``."""

    @property
    def ok(self) -> bool:
        return True


def validate(
    df: pl.DataFrame,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    on_error: str = "raise",
) -> tuple[pl.DataFrame, ValidationReport]:
    """Validate a long-format series frame against the freecast data contract.

    Parameters
    ----------
    df:
        A Polars DataFrame with at least ``unique_id``, ``ds``, ``y`` columns.
    min_history:
        Minimum number of observations a series must have to be usable.
    on_error:
        ``"raise"`` (default) aborts on the first violation with a specific
        ``ContractError``. ``"drop"`` removes offending series and returns a
        report describing what was dropped, keeping everything else usable.

    Returns
    -------
    (clean_df, report):
        The validated (and possibly filtered) frame, plus a report of what
        happened.
    """
    if on_error not in ("raise", "drop"):
        raise ValueError(f"on_error must be 'raise' or 'drop', got {on_error!r}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ContractError(
            f"Missing required column(s): {missing_cols}. "
            f"Expected at least {REQUIRED_COLUMNS}, got {df.columns}."
        )

    if df.height == 0:
        raise ContractError("Input frame is empty: no rows to forecast.")

    dropped: dict[str, str] = {}
    working = df

    # non-numeric y
    if not working.schema["y"].is_numeric():
        try:
            working = working.with_columns(pl.col("y").cast(pl.Float64, strict=True))
        except pl.exceptions.InvalidOperationError as exc:
            raise ContractError(
                f"Column 'y' contains non-numeric values and cannot be safely cast to float: {exc}"
            ) from exc
    else:
        working = working.with_columns(pl.col("y").cast(pl.Float64))

    # ds must be a date/datetime type (or parseable)
    if working.schema["ds"] not in (pl.Date, pl.Datetime):
        try:
            working = working.with_columns(pl.col("ds").str.to_date(strict=True))
        except Exception:
            try:
                working = working.with_columns(pl.col("ds").str.to_datetime(strict=True))
            except Exception as exc:
                raise ContractError(
                    f"Column 'ds' could not be parsed as a date/datetime. "
                    f"Got dtype {df.schema['ds']}. Error: {exc}"
                ) from exc

    # null values in required columns
    null_counts = working.select([pl.col(c).null_count().alias(c) for c in REQUIRED_COLUMNS]).row(
        0, named=True
    )
    null_cols = [c for c, n in null_counts.items() if n > 0]
    if null_cols:
        raise ContractError(
            f"Null values found in required column(s): {null_cols}. "
            "freecast does not silently impute missing identifiers, "
            "timestamps, or targets."
        )

    # duplicate (unique_id, ds) timestamps
    dupe_keys = (
        working.group_by(["unique_id", "ds"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    if dupe_keys.height > 0:
        offending_ids = set(dupe_keys["unique_id"].to_list())
        if on_error == "raise":
            example = dupe_keys.head(5).to_dicts()
            raise ContractError(
                f"Duplicate (unique_id, ds) timestamps found for "
                f"{len(offending_ids)} series. Examples: {example}. "
                "Each series must have exactly one observation per timestamp."
            )
        for uid in offending_ids:
            dropped[uid] = "duplicate timestamps for one or more dates"
        working = working.filter(~pl.col("unique_id").is_in(list(offending_ids)))

    # missing dates within a series' own range (irregular / gappy series)
    gap_ids = _find_date_gaps(working)
    if gap_ids:
        if on_error == "raise":
            raise ContractError(
                f"Missing dates detected within the observed range for "
                f"{len(gap_ids)} series (gaps in an otherwise regular "
                f"frequency), e.g. {sorted(gap_ids)[:5]}. Fill or resample "
                "before ingesting, or pass on_error='drop'."
            )
        for uid in gap_ids:
            dropped[uid] = "missing dates within observed range (irregular frequency)"
        working = working.filter(~pl.col("unique_id").is_in(list(gap_ids)))

    # insufficient history
    counts = working.group_by("unique_id").agg(pl.len().alias("n_obs"))
    short = counts.filter(pl.col("n_obs") < min_history)
    if short.height > 0:
        short_ids = set(short["unique_id"].to_list())
        if on_error == "raise":
            raise ContractError(
                f"{len(short_ids)} series have fewer than min_history="
                f"{min_history} observations, e.g. {sorted(short_ids)[:5]}. "
                "Increase history, lower min_history, or pass on_error='drop'."
            )
        for uid in short_ids:
            dropped[uid] = f"fewer than min_history={min_history} observations"
        working = working.filter(~pl.col("unique_id").is_in(list(short_ids)))

    if working.height == 0:
        raise ContractError("All series were dropped during validation; nothing left to forecast.")

    working = working.sort(["unique_id", "ds"])
    report = ValidationReport(
        n_series=working["unique_id"].n_unique(),
        n_rows=working.height,
        dropped_series=dropped,
    )
    return working, report


_GAP_TOLERANCE = 1.5


def _find_date_gaps(df: pl.DataFrame) -> set[str]:
    """Detect series with missing dates given their own inferred frequency.

    Uses each series' median gap between consecutive observations as its
    inferred step, then flags a series only if some gap exceeds
    ``_GAP_TOLERANCE`` times that median. The tolerance absorbs natural
    calendar variation (28-31 day months, 365/366 day years) without
    treating it as a missing observation, while still catching an actually
    skipped period (e.g. a ~2x gap).
    """
    gap_ids: set[str] = set()
    diffs = (
        df.sort(["unique_id", "ds"])
        .with_columns(pl.col("ds").diff().over("unique_id").alias("_gap"))
        .drop_nulls("_gap")
    )
    if diffs.height == 0:
        return gap_ids

    medians = diffs.group_by("unique_id").agg(pl.col("_gap").median().alias("_median_gap"))
    joined = diffs.join(medians, on="unique_id", how="left")
    offenders = joined.filter(pl.col("_gap") > pl.col("_median_gap") * _GAP_TOLERANCE)
    if offenders.height > 0:
        gap_ids = set(offenders["unique_id"].to_list())
    return gap_ids
