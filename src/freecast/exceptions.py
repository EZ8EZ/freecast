"""Exception reporting: which series actually need a planner's attention.

At thousands-of-SKU scale nobody can review every forecast. Forecast Pro
sells "exception reporting" as a named feature for exactly this reason.
freecast surfaces the same idea as a data query, not a dashboard (headless,
per the project's design): compute a handful of rule-based flags per series
and return the ones that tripped at least one, ranked by how many flags
fired, so a planner's limited review time goes where it matters.

Every flag is independently interpretable and independently thresholded —
there's no hidden scoring model. That's a deliberate choice: a planner
challenged on "why is this flagged" should get a specific, defensible
answer, not "the model said so."
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DEFAULT_ACCURACY_THRESHOLD = 1.0
"""For MASE/RMSSE, 1.0 means "no better than a naive forecast" — the
standard interpretation of those scaled metrics."""

DEFAULT_INTERVAL_WIDTH_THRESHOLD = 0.5
"""Flag when a forecast's (hi - lo) prediction interval exceeds this
fraction of the point forecast — a wide band relative to the number itself,
i.e. freecast is telling you it isn't confident."""

DEFAULT_JUMP_THRESHOLD = 0.5
"""Flag when the forecast's mean over the horizon differs from the series'
last known actual by more than this fraction."""

_SCALED_METRICS = {"mase", "rmsse"}


@dataclass
class ExceptionReport:
    flagged: pl.DataFrame
    """unique_id, flags (list[str]), n_flags, plus the diagnostic columns
    each flag was computed from — sorted by n_flags descending."""

    summary: dict[str, int]
    """flag name -> how many series tripped it."""


def find_exceptions(
    train_df: pl.DataFrame,
    forecasts: pl.DataFrame,
    selection: pl.DataFrame,
    *,
    metric: str = "mase",
    accuracy_threshold: float = DEFAULT_ACCURACY_THRESHOLD,
    level: int = 80,
    interval_width_threshold: float = DEFAULT_INTERVAL_WIDTH_THRESHOLD,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    prior_classification: pl.DataFrame | None = None,
    classification: pl.DataFrame | None = None,
) -> ExceptionReport:
    """Flag series worth a planner's review.

    Parameters
    ----------
    train_df: the (unique_id, ds, y) history the forecast was built from —
        used for the "large jump vs. last actual" flag.
    forecasts: a freecast forecasts frame (unique_id, ds, y_hat, lo-<level>,
        hi-<level>, ...).
    selection: a freecast selection frame (unique_id, model, <metric>).
    metric: which column of ``selection`` to read for the accuracy flag;
        only meaningful (and only applied) for "mase"/"rmsse", where 1.0 is
        the natural "no better than naive" threshold.
    level: which prediction-interval level to use for the width flag.
    prior_classification / classification: optional demand-type
        classifications (freecast.intermittent output) from a previous and
        current run — if both given, flags series whose category changed.
    """
    flag_frames: list[pl.DataFrame] = []
    diagnostics: list[pl.DataFrame] = []

    if metric in _SCALED_METRICS and metric in selection.columns:
        poor_accuracy = selection.filter(pl.col(metric) > accuracy_threshold).select(
            ["unique_id", pl.col(metric).alias("accuracy_metric")]
        )
        flag_frames.append(
            poor_accuracy.select(["unique_id", pl.lit("poor_accuracy").alias("flag")])
        )
        diagnostics.append(poor_accuracy)

    lo_col, hi_col = f"lo-{level}", f"hi-{level}"
    if lo_col in forecasts.columns and hi_col in forecasts.columns:
        widths = forecasts.with_columns(
            (
                (pl.col(hi_col) - pl.col(lo_col)) / pl.col("y_hat").abs().clip(lower_bound=1e-9)
            ).alias("max_relative_interval_width")
        )
        per_series_width = widths.group_by("unique_id").agg(
            pl.col("max_relative_interval_width").max()
        )
        wide = per_series_width.filter(
            pl.col("max_relative_interval_width") > interval_width_threshold
        )
        flag_frames.append(wide.select(["unique_id", pl.lit("wide_interval").alias("flag")]))
        diagnostics.append(wide)

    last_actual = (
        train_df.sort(["unique_id", "ds"])
        .group_by("unique_id")
        .agg(pl.col("y").last().alias("last_actual"))
    )
    mean_forecast = forecasts.group_by("unique_id").agg(
        pl.col("y_hat").mean().alias("mean_forecast")
    )
    jump = mean_forecast.join(last_actual, on="unique_id").with_columns(
        (
            (pl.col("mean_forecast") - pl.col("last_actual")).abs()
            / pl.col("last_actual").abs().clip(lower_bound=1e-9)
        ).alias("relative_jump")
    )
    jumped = jump.filter(pl.col("relative_jump") > jump_threshold)
    flag_frames.append(jumped.select(["unique_id", pl.lit("large_jump").alias("flag")]))
    diagnostics.append(jumped)

    if prior_classification is not None and classification is not None:
        changed = (
            prior_classification.select(["unique_id", pl.col("category").alias("prior_category")])
            .join(
                classification.select(["unique_id", pl.col("category").alias("new_category")]),
                on="unique_id",
            )
            .filter(pl.col("prior_category") != pl.col("new_category"))
        )
        flag_frames.append(
            changed.select(["unique_id", pl.lit("demand_type_changed").alias("flag")])
        )
        diagnostics.append(changed)

    if not any(f.height > 0 for f in flag_frames):
        empty = pl.DataFrame(
            schema={"unique_id": pl.Utf8, "flags": pl.List(pl.Utf8), "n_flags": pl.Int64}
        )
        return ExceptionReport(flagged=empty, summary={})

    all_flags = pl.concat(flag_frames, how="vertical")
    summary = {
        row["flag"]: row["n"]
        for row in all_flags.group_by("flag").agg(pl.len().alias("n")).to_dicts()
    }

    grouped = all_flags.group_by("unique_id").agg(
        pl.col("flag").unique().sort().alias("flags"), pl.col("flag").n_unique().alias("n_flags")
    )

    result = grouped
    for d in diagnostics:
        if d.height == 0:
            continue
        extra_cols = [c for c in d.columns if c != "unique_id"]
        result = result.join(d.select(["unique_id", *extra_cols]), on="unique_id", how="left")

    result = result.sort("n_flags", descending=True)
    return ExceptionReport(flagged=result, summary=summary)
