"""Runs the freecast engine against M3 / M4 / Tourism and reports accuracy.

Usage: ``freecast bench m3`` (or ``python -m bench.runner m3``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from bench.published_results import M3_OVERALL
from freecast.engine import ForecastEngine

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

# (horizon, freq, season_length) per M3 group, matching the original
# competition's own forecast horizons.
M3_GROUPS = {
    "Yearly": (6, "1y", 1),
    "Quarterly": (8, "1q", 4),
    "Monthly": (18, "1mo", 12),
    "Other": (8, "1d", 1),
}


@dataclass
class BenchSummary:
    dataset: str
    group: str
    n_series: int
    n_dropped: int
    smape: float
    mase: float
    seconds_elapsed: float
    model_counts: dict[str, int]


def _load_m3(group: str, data_dir: Path) -> pl.DataFrame:
    from datasetsforecast.m3 import M3

    df, *_ = M3.load(str(data_dir), group)
    return pl.from_pandas(df)


def _smape(y_true, y_hat) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    denom = np.abs(y_true) + np.abs(y_hat)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(y_true - y_hat) / denom)
    return float(100 * ratio.mean())


def _mase(y_true, y_hat, y_train, season_length: int) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    naive_errors = (
        np.abs(np.diff(y_train, n=1))
        if season_length <= 1
        else np.abs(y_train[season_length:] - y_train[:-season_length])
    )
    scale = naive_errors.mean() if naive_errors.size > 0 else np.nan
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(np.abs(y_true - y_hat).mean() / scale)


def run_m3_group(group: str, data_dir: Path = DEFAULT_DATA_DIR) -> BenchSummary:
    if group not in M3_GROUPS:
        raise ValueError(f"Unknown M3 group {group!r}; choose one of {sorted(M3_GROUPS)}")
    h, freq, season_length = M3_GROUPS[group]

    full = _load_m3(group, data_dir)
    train_parts, test_parts = [], []
    for _, series in full.group_by("unique_id"):
        series = series.sort("ds")
        train_parts.append(series.head(series.height - h))
        test_parts.append(series.tail(h))
    train_df = pl.concat(train_parts)
    test_df = pl.concat(test_parts)

    t0 = time.time()
    engine = ForecastEngine(h=h, freq=freq, n_windows=2, min_history=h + 2, on_error="drop")
    result = engine.run(train_df)
    elapsed = time.time() - t0

    joined = result.forecasts.join(test_df, on=["unique_id", "ds"], how="inner")
    smape = _smape(joined["y"].to_numpy(), joined["y_hat"].to_numpy())

    def _key(uid: object) -> object:
        return uid[0] if isinstance(uid, tuple) else uid

    mase_values = []
    train_by_id = {_key(uid): grp["y"].to_numpy() for uid, grp in train_df.group_by("unique_id")}
    for uid, grp in joined.group_by("unique_id"):
        m = _mase(
            grp["y"].to_numpy(), grp["y_hat"].to_numpy(), train_by_id[_key(uid)], season_length
        )
        if m == m:  # not NaN
            mase_values.append(m)
    mase = sum(mase_values) / len(mase_values) if mase_values else float("nan")

    model_counts = result.selection.group_by("model").agg(pl.len().alias("n")).to_dicts()
    model_counts_dict = {row["model"]: row["n"] for row in model_counts}

    return BenchSummary(
        dataset="m3",
        group=group,
        n_series=result.validation.n_series,
        n_dropped=len(result.validation.dropped_series),
        smape=smape,
        mase=mase,
        seconds_elapsed=elapsed,
        model_counts=model_counts_dict,
    )


def run_benchmark(dataset: str, group: str | None, output_dir: Path) -> str:
    dataset = dataset.lower()
    if dataset != "m3":
        raise NotImplementedError(
            f"Benchmark for {dataset!r} is not wired up yet; only 'm3' is available today."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [group] if group else list(M3_GROUPS)

    summaries = []
    lines = ["freecast vs. published M3 competition results (overall sMAPE / MASE):"]
    for name, (m_smape, m_mase) in M3_OVERALL.items():
        lines.append(f"  {name:<28} sMAPE={m_smape:>6.2f}  MASE={m_mase:>5.2f}")
    lines.append("")

    for g in groups:
        summary = run_m3_group(g)
        summaries.append(summary)
        lines.append(
            f"freecast / M3-{g:<10} n={summary.n_series:<5} (dropped {summary.n_dropped:<3}) "
            f"sMAPE={summary.smape:>6.2f}  MASE={summary.mase:>5.2f}  "
            f"({summary.seconds_elapsed:.1f}s)  models={summary.model_counts}"
        )

    if len(summaries) > 1:
        total_n = sum(s.n_series for s in summaries)
        weighted_smape = sum(s.smape * s.n_series for s in summaries) / total_n
        weighted_mase = sum(s.mase * s.n_series for s in summaries) / total_n
        lines.append(
            f"freecast / M3-{'Overall':<10} n={total_n:<5} "
            f"sMAPE={weighted_smape:>6.2f}  MASE={weighted_mase:>5.2f}"
        )

    out_path = output_dir / "m3_summary.json"
    out_path.write_text(json.dumps([asdict(s) for s in summaries], indent=2))
    lines.append(f"\nWrote {out_path}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "m3"
    print(run_benchmark(dataset, None, Path("bench/results")))
