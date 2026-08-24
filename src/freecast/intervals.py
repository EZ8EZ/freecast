"""Conformal prediction intervals for every forecast freecast produces.

Wraps ``statsforecast.utils.ConformalIntervals``, which builds distribution-
free prediction intervals from the empirical distribution of cross-validated
residuals rather than assuming Gaussian errors. This is applied uniformly
across all models (ETS, ARIMA, Theta, CES, Croston-family) so no forecast
leaves the engine as a point estimate alone.
"""

from __future__ import annotations

from statsforecast.utils import ConformalIntervals

DEFAULT_LEVELS = (80, 95)


def build_conformal_intervals(h: int, n_windows: int = 2) -> ConformalIntervals:
    """Build a conformal-interval spec for the given horizon.

    ``n_windows`` controls how many backtest windows are used to build the
    empirical residual distribution; 2 is statsforecast's own default and a
    reasonable floor for short series.
    """
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    return ConformalIntervals(h=h, n_windows=n_windows, method="conformal_distribution")
