"""Optional integration with The Forecasting Company's t0 zero-shot foundation model.

t0 (Apache-2.0, https://github.com/theforecastingcompany/tfc-t0) is a
~102M-parameter decoder transformer that produces zero-shot probabilistic
forecasts with no per-series training or backtesting required. It
complements freecast's CV-selected statistical pool especially for series
too short to backtest reliably.

This integration is entirely optional:

- Requires the ``foundation`` extra: ``pip install "freecast[foundation]"``.
- Requires access to the gated ``theforecastingcompany/t0-alpha`` Hugging
  Face repo (request access on the model page, then ``hf auth login``).
  freecast does not bundle or auto-download weights.
- t0 only ever emits five fixed quantiles (0.1, 0.25, 0.5, 0.75, 0.9), so it
  can only serve prediction-interval levels derivable from that set: 80
  (from the 0.1/0.9 pair) and 50 (from 0.25/0.75). Any other requested level
  makes it ineligible as a candidate for that run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

MODEL_NAME = "T0"
DEFAULT_PRETRAINED = "theforecastingcompany/t0-alpha"
DEFAULT_CONTEXT_LENGTH = 512

# level -> (lower quantile, upper quantile), constrained by t0's fixed quantile set.
SUPPORTED_LEVELS = {80: (0.1, 0.9), 50: (0.25, 0.75)}


class FoundationModelUnavailable(RuntimeError):
    """Raised when tfc-t0/torch aren't installed, or weights can't be fetched."""


def _import_t0() -> tuple[Any, Any]:
    try:
        import torch
        from t0 import T0Forecaster
    except ImportError as exc:
        raise FoundationModelUnavailable(
            "The 't0' foundation model requires the optional 'foundation' extra: "
            'pip install "freecast[foundation]". It also needs access to the '
            "gated theforecastingcompany/t0-alpha Hugging Face repo (request "
            "access on the model page, then run `hf auth login`)."
        ) from exc
    return torch, T0Forecaster


def levels_supported(levels: tuple[int, ...]) -> bool:
    """True if every requested prediction-interval level is derivable from t0's quantiles."""
    return all(lv in SUPPORTED_LEVELS for lv in levels)


@dataclass
class T0Model:
    """Thin wrapper making t0 batch-callable like the rest of freecast's model pool."""

    forecaster: Any

    @classmethod
    def load(cls, pretrained: str = DEFAULT_PRETRAINED) -> T0Model:
        """Download (if needed) and load the pretrained t0 weights.

        Raises FoundationModelUnavailable if torch/tfc-t0 aren't installed,
        or if the weights can't be fetched (e.g. no Hugging Face access to
        the gated repo).
        """
        torch, T0Forecaster = _import_t0()
        try:
            forecaster = T0Forecaster.from_pretrained(pretrained).eval()
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            raise FoundationModelUnavailable(
                f"Could not load t0 weights from {pretrained!r}: {exc}"
            ) from exc
        return cls(forecaster=forecaster)

    def predict(
        self,
        df: pl.DataFrame,
        *,
        h: int,
        freq: str | int,
        levels: tuple[int, ...] = (80,),
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> pl.DataFrame:
        """Zero-shot batch forecast.

        ``df`` is a long-format (unique_id, ds, y) frame. Returns a wide
        frame: unique_id, ds, T0, T0-lo-<level>, T0-hi-<level>.
        """
        if not levels_supported(levels):
            raise ValueError(
                f"t0 only supports prediction-interval levels {sorted(SUPPORTED_LEVELS)} "
                f"(derived from its fixed quantile set); got {levels}."
            )
        torch, _ = _import_t0()

        series = list(df.sort(["unique_id", "ds"]).group_by("unique_id", maintain_order=True))
        uids = [str(uid[0]) for uid, _ in series]
        contexts = np.zeros((len(series), context_length), dtype=np.float32)
        last_dates = []
        for i, (_, grp) in enumerate(series):
            y = grp["y"].to_numpy().astype(np.float32)
            tail = y[-context_length:]
            contexts[i, -len(tail) :] = tail
            last_dates.append(grp["ds"].max())

        quantiles_needed = sorted({0.5} | {q for lv in levels for q in SUPPORTED_LEVELS[lv]})
        with torch.no_grad():
            out = self.forecaster.predict(
                torch.from_numpy(contexts), horizon=h, quantiles=quantiles_needed
            )
        q_arr = np.asarray(out.quantiles)  # (batch, horizon, n_quantiles)
        q_idx = {q: quantiles_needed.index(q) for q in quantiles_needed}

        rows: list[dict[str, Any]] = []
        for i, uid in enumerate(uids):
            future_dates = pd.date_range(start=last_dates[i], periods=h + 1, freq=freq)[1:]
            for t in range(h):
                row: dict[str, Any] = {
                    "unique_id": uid,
                    "ds": future_dates[t],
                    MODEL_NAME: float(q_arr[i, t, q_idx[0.5]]),
                }
                for lv in levels:
                    lo_q, hi_q = SUPPORTED_LEVELS[lv]
                    row[f"{MODEL_NAME}-lo-{lv}"] = float(q_arr[i, t, q_idx[lo_q]])
                    row[f"{MODEL_NAME}-hi-{lv}"] = float(q_arr[i, t, q_idx[hi_q]])
                rows.append(row)

        result = pl.DataFrame(rows)
        return result.with_columns(pl.col("ds").cast(df.schema["ds"]))
