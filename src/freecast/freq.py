"""Canonical frequency handling.

freecast accepts both pandas-style frequency aliases (``"MS"``, ``"D"``,
``"W"``, ``"Q"``...) and Polars-style offset strings (``"1mo"``, ``"1d"``,
``"1w"``, ``"1q"``...), since callers reasonably expect either depending on
which ecosystem they're coming from.

Internally these need to reach three different consumers in three different
dialects:

- ``statsforecast`` operating on a Polars frame requires a Polars offset
  string (e.g. ``"1mo"``), and rejects pandas-style aliases outright.
- ``pandas.date_range`` (used by the optional t0 foundation-model
  integration to compute future timestamps) requires a pandas-style alias,
  and does not understand Polars offset syntax.
- Model selection and interval logic need an integer seasonal period (e.g.
  12 for monthly data) inferred from the frequency's *unit*, regardless of
  which dialect it was spelled in.

``resolve_freq`` parses either dialect once and returns all three so the
rest of the codebase never has to re-derive them (or, worse, silently
mismatch them — see the season-length bug this replaced, where a Polars-style
freq like ``"1mo"`` failed to match any pandas-style lookup key and silently
fell back to a season length of 1 for every series).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit code -> (Polars offset suffix, pandas offset alias, default
# seasonal period for that unit). Keyed by the upper-cased alpha suffix of
# either dialect's frequency string, so both "MS" (pandas) and "mo"/"MO"
# (Polars) resolve to the same row.
_UNIT_TABLE: dict[str, tuple[str, str, int]] = {
    "Y": ("y", "YS", 1),
    "A": ("y", "YS", 1),
    "AS": ("y", "YS", 1),
    "YS": ("y", "YS", 1),
    "Q": ("q", "QS", 4),
    "QS": ("q", "QS", 4),
    "M": ("mo", "MS", 12),
    "MS": ("mo", "MS", 12),
    "MO": ("mo", "MS", 12),
    "W": ("w", "W", 52),
    "D": ("d", "D", 7),
    "B": ("d", "D", 5),
    "H": ("h", "h", 24),
    "T": ("m", "min", 1),
    "MIN": ("m", "min", 1),
    "S": ("s", "s", 1),
}

_FREQ_PATTERN = re.compile(r"^(\d*)([A-Za-z]+)$")


@dataclass(frozen=True)
class ResolvedFreq:
    polars: str | int
    """Frequency in the form statsforecast expects when operating on Polars frames."""

    pandas: str | int
    """Frequency in the form ``pandas.date_range`` expects."""

    season_length: int
    """Best-effort seasonal period inferred from the frequency's unit."""


def resolve_freq(freq: str | int) -> ResolvedFreq:
    """Parse a pandas- or Polars-style frequency into both dialects plus a season length.

    Raises ``ValueError`` for an unrecognized frequency string.
    """
    if isinstance(freq, int):
        return ResolvedFreq(polars=freq, pandas=freq, season_length=1)

    match = _FREQ_PATTERN.match(freq.strip())
    if match is None:
        raise ValueError(f"Unrecognized frequency string: {freq!r}")
    n_str, unit = match.groups()
    key = unit.upper()
    if key not in _UNIT_TABLE:
        raise ValueError(
            f"Unrecognized frequency alias {unit!r} in {freq!r}. "
            f"Supported units: {sorted(_UNIT_TABLE)}."
        )
    polars_unit, pandas_unit, season_length = _UNIT_TABLE[key]
    n = int(n_str) if n_str else 1

    polars_freq = f"{n}{polars_unit}"
    pandas_freq = pandas_unit if n == 1 else f"{n}{pandas_unit}"
    # A multi-step frequency (e.g. "2MS") doesn't have an obvious seasonal
    # period, so don't guess one.
    if n != 1:
        season_length = 1

    return ResolvedFreq(polars=polars_freq, pandas=pandas_freq, season_length=season_length)
