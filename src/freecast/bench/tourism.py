"""Loader and paper-exact metrics for the 2010 Tourism Forecasting Competition.

Data: the 1,311 series from Athanasopoulos, Hyndman, Song & Wu (2011), "The
tourism forecasting competition", International Journal of Forecasting
27(3), 822-844, as released in the original competition zip. The primary
host (robjhyndman.com) serves a bot-check page to scripted downloads, so the
loader falls back to a byte-identical copy kept in the Tcomp R package's
repository, pinned to one commit. Either way, the file is only accepted if
it matches the SHA-256 below.

Nixtla's current ``datasetsforecast`` has no loader for this dataset (its
``TourismSmall``/``TourismLarge`` are a different, hierarchical dataset), so
freecast parses the competition CSVs directly: one column per series, with
header rows for series length, start year, and (monthly/quarterly only)
start period, followed by the values. ``*_in.csv`` is the training data and
``*_oos.csv`` is the competition's hold-out, the last h observations.

Metrics reproduce the paper's published tables exactly, since the whole
point is comparing against them. MAPE is plain MAPE (not sMAPE). MASE
divides by the mean absolute seasonal-naive difference |y_t - y_{t-m}|
(m = 12/4/1) computed over the *whole* series, training and hold-out
together, not over the training data alone. That's unusual, but it is what
the published numbers used: recomputing the paper's own Naive/SNaive
benchmarks on this data gives MAPE 23.61/16.46/22.56 and MASE 2.50/1.59/1.54
(yearly/quarterly/monthly), matching Tables 4-6 to the digit, while the
training-only scale gives 3.01/1.70/1.63. The scale is the same for every
method, so the comparison stays fair. Both metrics average over the full
horizon and then across series (the paper's "Average 1-h" column).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

ZIP_NAME = "27-3-Athanasopoulos1.zip"
ZIP_SHA256 = "0d21e4efae1090849171586e598c743edd2ceef967274351647faa21807c6d99"
ZIP_URLS = (
    "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip",
    "https://raw.githubusercontent.com/ellisp/Tcomp-r-package/"
    "33503f9c4bb71c33b8617db64f5ae11bc9cc7e0e/source-data/27-3-Athansopoulos1.zip",
)

# group: (horizon, Polars freq, season length m), per the paper's Table 1.
TOURISM_GROUPS = {
    "Yearly": (4, "1y", 1),
    "Quarterly": (8, "1q", 4),
    "Monthly": (24, "1mo", 12),
}


def _fetch_zip(data_dir: Path) -> bytes:
    cached = data_dir / "tourism" / ZIP_NAME
    if cached.exists():
        payload = cached.read_bytes()
        if hashlib.sha256(payload).hexdigest() == ZIP_SHA256:
            return payload

    errors = []
    for url in ZIP_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URLs)
                payload = resp.read()
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest != ZIP_SHA256:
            errors.append(f"{url}: unexpected content (sha256 {digest[:12]}...)")
            continue
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
        return payload
    raise RuntimeError("Could not download the Tourism competition data:\n" + "\n".join(errors))


def _period_start(year: int, period: int, group: str) -> dt.date:
    if group == "Monthly":
        return dt.date(year, period, 1)
    if group == "Quarterly":
        return dt.date(year, 3 * (period - 1) + 1, 1)
    return dt.date(year, 1, 1)


def _parse(text: str, group: str, freq: str) -> pl.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    names = rows[0]
    n_header = 3 if group == "Yearly" else 4
    frames = []
    for j, name in enumerate(names):
        n = int(rows[1][j])
        year = int(rows[2][j])
        period = int(rows[3][j]) if group != "Yearly" else 1
        values = [float(r[j]) for r in rows[n_header : n_header + n]]
        start = _period_start(year, period, group)
        frames.append(
            pl.DataFrame(
                {
                    "unique_id": [name] * n,
                    "ds": pl.date_range(
                        start, _offset(start, freq, n - 1), interval=freq, eager=True
                    ),
                    "y": values,
                }
            )
        )
    return pl.concat(frames)


def _offset(start: dt.date, freq: str, steps: int) -> dt.date:
    months = {"1mo": 1, "1q": 3, "1y": 12}[freq] * steps
    total = start.year * 12 + (start.month - 1) + months
    return dt.date(total // 12, total % 12 + 1, 1)


def load_tourism(group: str, data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(train_df, test_df)`` long frames (unique_id, ds, y) for one group."""
    if group not in TOURISM_GROUPS:
        raise ValueError(f"Unknown Tourism group {group!r}; choose one of {sorted(TOURISM_GROUPS)}")
    _, freq, _ = TOURISM_GROUPS[group]
    with zipfile.ZipFile(io.BytesIO(_fetch_zip(data_dir))) as zf:
        prefix = group.lower()
        train = _parse(zf.read(f"{prefix}_in.csv").decode(), group, freq)
        test = _parse(zf.read(f"{prefix}_oos.csv").decode(), group, freq)

    # The hold-out always continues its own training series, so date it that
    # way rather than trusting the file's start-year header: Y18's header
    # repeats its last training year, which would overlap the two.
    next_start = train.group_by("unique_id").agg(pl.col("ds").max().alias("_end"))
    test = (
        test.sort(["unique_id", "ds"])
        .join(next_start, on="unique_id")
        .with_columns(pl.int_range(1, pl.len() + 1).over("unique_id").alias("_step"))
    )
    test = test.with_columns(
        pl.struct(["_end", "_step"])
        .map_elements(lambda r: _offset(r["_end"], freq, r["_step"]), return_dtype=pl.Date)
        .alias("ds")
    ).drop(["_end", "_step"])
    return train, test


def paper_mape(joined: pl.DataFrame) -> float:
    """Mean over series of the mean absolute percentage error over the horizon."""
    per_series = joined.group_by("unique_id").agg(
        ((pl.col("y") - pl.col("y_hat")).abs() / pl.col("y").abs()).mean().alias("ape")
    )
    return float(100 * np.mean(per_series["ape"].to_numpy()))


def paper_mase(
    joined: pl.DataFrame, train_df: pl.DataFrame, test_df: pl.DataFrame, m: int
) -> float:
    """Mean over series of MAE / mean |y_t - y_{t-m}| over the full series.

    The scale spans training *and* hold-out data, matching the paper's
    published tables (see the module docstring).
    """
    full = pl.concat(
        [train_df.select(["unique_id", "ds", "y"]), test_df.select(["unique_id", "ds", "y"])]
    )
    scales = {}
    for (uid,), grp in full.sort(["unique_id", "ds"]).group_by("unique_id"):
        y = grp["y"].to_numpy()
        scales[uid] = float(np.mean(np.abs(y[m:] - y[:-m])))
    per_series = joined.group_by("unique_id").agg(
        (pl.col("y") - pl.col("y_hat")).abs().mean().alias("mae")
    )
    values = [row["mae"] / scales[row["unique_id"]] for row in per_series.iter_rows(named=True)]
    return float(np.mean(values))


def naive_forecast(train_df: pl.DataFrame, test_df: pl.DataFrame, m: int) -> pl.DataFrame:
    """Seasonal naive (m > 1) or naive (m == 1): the paper's own benchmarks."""
    parts = []
    for (uid,), test in test_df.sort(["unique_id", "ds"]).group_by("unique_id"):
        last = train_df.filter(pl.col("unique_id") == uid).sort("ds")["y"].tail(m).to_numpy()
        h = test.height
        y_hat = np.resize(last, h) if m > 1 else np.repeat(last[-1], h)
        parts.append(test.with_columns(pl.Series("y_hat", y_hat)))
    return pl.concat(parts)
