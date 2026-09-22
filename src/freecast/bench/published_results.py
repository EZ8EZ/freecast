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

# Tourism forecasting competition: Athanasopoulos, G., Hyndman, R.J., Song, H.
# & Wu, D.C. (2011), "The tourism forecasting competition," International
# Journal of Forecasting 27(3), 822-844, Tables 4-6 ("Average 1-h" columns),
# https://robjhyndman.com/papers/forecompijf.pdf. "ForePro" is Forecast Pro,
# run by the authors as one of the benchmark methods. The paper reports
# SNaive for monthly/quarterly and Naive for yearly. freecast's harness
# reproduces the paper's Naive/SNaive rows exactly (see bench/tourism.py),
# which checks both the data and the metric definitions.
TOURISM = {
    # group: {method: (MAPE, MASE)}
    "Monthly": {
        "ForePro": (19.91, 1.40),
        "ETS": (21.15, 1.49),
        "ARIMA": (21.13, 1.38),
        "Theta": (22.11, 1.55),
        "Damped": (23.47, 1.66),
        "SNaive": (22.56, 1.54),
    },
    "Quarterly": {
        "ForePro": (15.72, 1.48),
        "ETS": (16.05, 1.58),
        "ARIMA": (16.23, 1.47),
        "Theta": (16.15, 1.56),
        "Damped": (15.56, 1.43),
        "SNaive": (16.46, 1.59),
    },
    "Yearly": {
        "ForePro": (26.36, 2.65),
        "ETS": (27.68, 2.71),
        "ARIMA": (28.03, 2.63),
        "Theta": (23.45, 2.28),
        "Damped": (28.15, 2.75),
        "Naive": (23.61, 2.50),
    },
}
