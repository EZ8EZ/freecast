"""Hierarchical reconciliation: coherent forecasts across a product/region/etc. hierarchy.

Planners don't forecast one SKU in isolation — they need bottom-level
forecasts to sum coherently into category/region/company totals for S&OP
and finance review. Forecasting every level independently produces
incoherent totals (the region total doesn't equal the sum of its SKUs);
reconciliation is the standard fix.

This wraps Nixtla's ``hierarchicalforecast`` (Apache-2.0, same ecosystem as
``statsforecast``): aggregate the bottom-level series up through a defined
hierarchy, forecast every level with the existing engine (each level is just
another synthetic series to it — no separate forecasting logic needed), then
reconcile with MinTrace by default, the modern standard method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, HReconciler, MinTrace
from hierarchicalforecast.utils import aggregate

from freecast.engine import ForecastEngine, ForecastResult

DEFAULT_RECONCILER_NAME = "y_hat/MinTrace_method-wls_struct"
BOTTOM_UP_RECONCILER_NAME = "y_hat/BottomUp"


def default_reconcilers() -> list[HReconciler]:
    # wls_struct scales by each series' number of bottom-level descendants —
    # a standard, well-behaved default that needs only actuals. The
    # residual-based methods (wls_var, mint_cov, mint_shrink) are more
    # accurate in principle but need in-sample fitted values the engine
    # doesn't currently expose; pass custom `reconcilers=` with those methods
    # plus fitted values in `Y_df` if you need them.
    return [BottomUp(), MinTrace(method="wls_struct")]


@dataclass
class HierarchyResult:
    forecasts: pl.DataFrame
    """unique_id, ds, y_hat (unreconciled), plus one column per reconciler
    (e.g. y_hat/BottomUp, y_hat/MinTrace_method-wls_struct) — see
    DEFAULT_RECONCILER_NAME / BOTTOM_UP_RECONCILER_NAME for the default names."""

    engine_result: ForecastResult
    """The underlying per-level ForecastEngine output, for selection/FVA/audit purposes."""

    aggregated: pl.DataFrame
    """The full (unique_id, ds, y) series at every hierarchy level, bottom to top."""

    tags: dict[str, object]
    """level name -> array of unique_ids at that level, as returned by ``aggregate()``."""

    s_matrix: pl.DataFrame
    """The summing matrix mapping bottom-level series to every aggregate, for inspection."""


def build_hierarchy(
    df: pl.DataFrame, spec: list[list[str]]
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Aggregate a bottom-level (grouping_cols..., ds, y) frame up through ``spec``.

    ``spec`` is a list of grouping-column combinations from the top level down
    to the full bottom level, e.g. ``[["category"], ["category", "region"]]``
    aggregates first by category alone, then by category+region — the bottom
    level (every grouping column together) is included automatically.
    """
    if not spec:
        raise ValueError("spec must list at least one level of grouping columns.")
    return aggregate(df=df, spec=spec)


def reconcile(
    df: pl.DataFrame,
    spec: list[list[str]],
    *,
    h: int,
    freq: str | int,
    reconcilers: list[HReconciler] | None = None,
    **engine_kwargs: Any,
) -> HierarchyResult:
    """Forecast and reconcile a hierarchy end to end.

    Parameters
    ----------
    df: bottom-level long-format frame with the grouping columns from
        ``spec``, plus ``ds`` and ``y``.
    spec: hierarchy levels, top to bottom — see ``build_hierarchy``.
    h, freq, **engine_kwargs: passed straight through to ``ForecastEngine``.
    reconcilers: hierarchicalforecast reconciler instances; defaults to
        BottomUp (naive baseline) and MinTrace with shrinkage (the modern
        standard).
    """
    aggregated, s_df, tags = build_hierarchy(df, spec)

    engine = ForecastEngine(h=h, freq=freq, **engine_kwargs)
    engine_result = engine.run(aggregated)

    y_hat_df = engine_result.forecasts.select(["unique_id", "ds", "y_hat"])

    hrec = HierarchicalReconciliation(reconcilers=reconcilers or default_reconcilers())
    reconciled = hrec.reconcile(Y_hat_df=y_hat_df, Y_df=aggregated, S_df=s_df, tags=tags)

    return HierarchyResult(
        forecasts=reconciled,
        engine_result=engine_result,
        aggregated=aggregated,
        tags=tags,
        s_matrix=s_df,
    )


def check_coherence(
    forecasts: pl.DataFrame, s_matrix: pl.DataFrame, value_col: str, *, atol: float = 1e-6
) -> bool:
    """Verify that every aggregate level's forecast equals the sum of its bottom-level children.

    Mostly useful in tests: a passing reconciliation guarantees this by
    construction, so a failure here means the reconciler/summing-matrix
    inputs were mismatched, not a numerical fluke.
    """
    bottom_ids = [c for c in s_matrix.columns if c != "unique_id"]
    bottom = forecasts.filter(pl.col("unique_id").is_in(bottom_ids)).select(
        ["unique_id", "ds", value_col]
    )
    wide = bottom.pivot(on="unique_id", index="ds", values=value_col).sort("ds")

    for row in s_matrix.iter_rows(named=True):
        agg_id = row["unique_id"]
        weights = {k: v for k, v in row.items() if k != "unique_id" and v != 0}
        expected = None
        for bottom_id, weight in weights.items():
            term = wide[bottom_id] * weight
            expected = term if expected is None else expected + term
        actual = forecasts.filter(pl.col("unique_id") == agg_id).sort("ds")[value_col]
        if expected is None or not (actual - expected).abs().max() <= atol:
            return False
    return True
