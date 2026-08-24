"""Intermittent-demand detection via the Syntetos-Boylan ADI / CV^2 classification.

Regular ETS/ARIMA models assume smooth, continuously-demanded series. Sparse
or lumpy demand breaks that assumption, so series are classified first and
routed to Croston-family models when appropriate.

Classification (Syntetos & Boylan, 2005):
    ADI  = average inter-demand interval (mean gap between non-zero periods)
    CV^2 = squared coefficient of variation of the non-zero demand sizes

    ADI < 1.32, CV^2 < 0.49   -> smooth       (regular models)
    ADI >= 1.32, CV^2 < 0.49  -> intermittent (Croston-family)
    ADI < 1.32, CV^2 >= 0.49  -> erratic      (Croston-family)
    ADI >= 1.32, CV^2 >= 0.49 -> lumpy        (Croston-family)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


@dataclass(frozen=True)
class DemandClassification:
    unique_id: str
    adi: float
    cv2: float
    category: str

    @property
    def is_intermittent(self) -> bool:
        return self.category != "smooth"


def classify_series(df: pl.DataFrame) -> pl.DataFrame:
    """Classify each series in a long-format (unique_id, ds, y) frame.

    Returns a DataFrame with columns: unique_id, adi, cv2, category.
    A series with fewer than two non-zero demand periods cannot have a
    meaningful CV^2 and is classified as "smooth" (falls back to regular
    models) unless it is entirely zero, in which case it is "intermittent".
    """
    rows: list[dict] = []
    for uid, group in df.sort(["unique_id", "ds"]).group_by("unique_id", maintain_order=True):
        y = group["y"].to_numpy()
        rows.append(_classify_one(str(uid[0]), y))
    return pl.DataFrame(rows)


def _classify_one(unique_id: str, y) -> dict:
    import numpy as np

    y = np.asarray(y, dtype=float)
    nonzero_mask = y != 0
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero == 0:
        return {"unique_id": unique_id, "adi": float("inf"), "cv2": 0.0, "category": "intermittent"}

    if n_nonzero < 2:
        # Not enough non-zero points to estimate variability; treat as smooth
        # so it falls through to the standard model pool rather than forcing
        # a degenerate Croston fit.
        return {"unique_id": unique_id, "adi": 1.0, "cv2": 0.0, "category": "smooth"}

    nonzero_idx = np.flatnonzero(nonzero_mask)
    intervals = np.diff(nonzero_idx)
    adi = float(intervals.mean()) if intervals.size > 0 else 1.0

    demand_sizes = y[nonzero_mask]
    mean_size = float(demand_sizes.mean())
    std_size = float(demand_sizes.std(ddof=1))
    cv2 = (std_size / mean_size) ** 2 if mean_size > 0 else 0.0

    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        category = "smooth"
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        category = "intermittent"
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        category = "erratic"
    else:
        category = "lumpy"

    return {"unique_id": unique_id, "adi": adi, "cv2": cv2, "category": category}


def split_by_demand_type(
    df: pl.DataFrame, classification: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a series frame into (regular, intermittent) sub-frames.

    "Regular" series are classified "smooth"; everything else (intermittent,
    erratic, lumpy) is routed to Croston-family models.
    """
    if classification is None:
        classification = classify_series(df)

    regular_ids = classification.filter(pl.col("category") == "smooth")["unique_id"].to_list()
    intermittent_ids = classification.filter(pl.col("category") != "smooth")["unique_id"].to_list()

    regular_df = df.filter(pl.col("unique_id").is_in(regular_ids))
    intermittent_df = df.filter(pl.col("unique_id").is_in(intermittent_ids))
    return regular_df, intermittent_df
