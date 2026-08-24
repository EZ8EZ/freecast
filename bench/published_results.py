"""Published competition benchmark numbers, for side-by-side comparison only.

These are NOT computed by freecast — they are reference figures from prior
published work, reproduced here so `freecast bench` can print them next to
our own numbers. Sources:

- M3: overall-average sMAPE/MASE across all 3,003 M3 series, as tabulated in
  Rob Hyndman's Mcomp package documentation
  (https://pkg.robjhyndman.com/Mcomp/articles/Comparisons.html), which itself
  reproduces the original competition results from Makridakis, S. & Hibon, M.
  (2000), "The M3-Competition: results, conclusions and implications,"
  International Journal of Forecasting, 16(4), 451-476. ForecastPro was a
  named commercial entrant in the M3 competition.
"""

from __future__ import annotations

M3_OVERALL = {
    # method: (sMAPE, MASE)
    "Theta": (13.01, 1.39),
    "ForecastPro": (13.19, 1.47),
    "ForecastX": (13.49, 1.42),
    "ETS": (13.07, 1.43),
    "AutoARIMA": (13.57, 1.45),
}
