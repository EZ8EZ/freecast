# freecast

A headless, open-source batch time-series forecasting engine — the cheap,
uncluttered alternative to Forecast Pro.

## Why this exists

The math behind classical forecasting — ETS, ARIMA, Theta, Croston — has
been commoditized. Permissively-licensed, genuinely fast implementations
(Nixtla's `statsforecast`) exist for all of it. What legacy commercial tools
like Forecast Pro actually sell isn't the statistics; it's the workflow
around it: an expert system that picks a model per series instead of forcing
one method on everything, batch processing across thousands of SKUs, and
prediction intervals a planner can defend.

`freecast` clones the math for free and puts the effort into that workflow
layer instead:

- **Cross-validation-driven model selection**, not a hardcoded rule table.
  Every series is backtested (rolling-origin CV) across a candidate pool —
  AutoETS, AutoARIMA, AutoTheta, AutoCES — and the winner is picked by an
  explicit, configurable accuracy metric (MASE by default).
- **Automatic intermittent-demand routing.** Sparse/lumpy series are
  detected via the standard Syntetos-Boylan ADI/CV² classification and
  routed to Croston-family models (Croston, CrostonSBA, TSB, ADIDA, IMAPA)
  instead of having ETS/ARIMA forced onto data that breaks their
  assumptions.
- **Conformal prediction intervals on every forecast.** No point forecast
  ships without a distribution-free, empirically-calibrated uncertainty
  band.
- **Batch scale.** Tens of thousands of independent series, one call,
  vectorized/parallel execution on a laptop.
- **Headless by design.** A library and a thin CLI on top of it. No GUI, no
  bundled dashboard — this is meant to be called from your own pipeline, or
  eventually from an MCP server (see Roadmap).

`freecast` is built on the [Nixtla](https://github.com/Nixtla) ecosystem
(`statsforecast`, `utilsforecast`), [Polars](https://pola.rs), and
[DuckDB](https://duckdb.org). All Apache-2.0.

## Install

```bash
pip install freecast
# or, for local development:
git clone https://github.com/EZ8EZ/freecast
cd freecast
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quickstart

### As a library

```python
import polars as pl
from freecast import ForecastEngine

df = pl.read_csv("sales.csv")  # columns: unique_id, ds, y

engine = ForecastEngine(h=12, freq="MS")  # 12-step horizon, monthly data
result = engine.run(df)

result.forecasts        # unique_id, ds, model, y_hat, lo-80, hi-80, lo-95, hi-95
result.selection         # unique_id, model, mase — the chosen model per series
result.classification    # unique_id, adi, cv2, category — demand-type routing
result.validation         # n_series, n_rows, dropped_series
```

### From the CLI

```bash
freecast run sales.csv --horizon 12 --freq MS --output-dir out/
```

Writes `forecasts.parquet`, `model_selection.parquet`, and
`demand_classification.parquet` to `out/`.

```
$ freecast run sales.csv --horizon 12 --freq MS
Forecasted 812 series (3 dropped) at horizon=12, freq='MS'.
Wrote forecasts, model_selection, demand_classification to freecast_output/
```

### The data contract

Input must be long-format with columns `unique_id`, `ds`, `y` (plus any
number of exogenous regressor columns, passed through untouched). Ingest
fails loudly and specifically rather than silently coercing bad data:
missing required columns, non-numeric `y`, null identifiers/timestamps/
targets, duplicate `(unique_id, ds)` pairs, gaps in an otherwise-regular
frequency, and series with too little history to forecast are all rejected
by default. Pass `on_error="drop"` (`--on-error drop` on the CLI) to instead
drop the offending series and keep going — the returned validation report
says exactly what was dropped and why.

## Benchmarks

`freecast` includes a reproducible benchmark harness against public
M-competition datasets — the same competitions Forecast Pro was a named
commercial entrant in. Run it yourself:

```bash
pip install "freecast[bench]"
freecast bench m3
```

M3 comprises 3,003 series across four frequencies (Yearly, Quarterly,
Monthly, Other), each forecast at the competition's own horizon (6, 8, 18,
and 8 steps respectively). Published reference numbers below are the
overall-average sMAPE/MASE across all 3,003 series, as tabulated in Rob
Hyndman's [Mcomp package
documentation](https://pkg.robjhyndman.com/Mcomp/articles/Comparisons.html),
reproducing the original results from Makridakis & Hibon (2000), *The
M3-Competition: results, conclusions and implications*, International
Journal of Forecasting 16(4). **Forecast Pro was a named commercial entrant**
in that competition — this is the actual, verifiable score we're comparing
against.

<!-- BENCH_RESULTS -->

Run `freecast bench m3` yourself to reproduce these numbers, or to break
them down by frequency group with `freecast bench m3 --group Monthly`.

M4 and Tourism harnesses are stubbed out in `bench/` for a follow-up; M5 is
out of scope for now given its size.

## Architecture

```
src/freecast/
├── contract.py       # data validation — fails loudly on bad input
├── intermittent.py   # ADI/CV² intermittent-demand classification
├── selection.py       # cross-validation-driven model selection
├── intervals.py       # conformal prediction interval builder
├── engine.py           # orchestrates the above into one call
└── cli.py               # the `freecast` command
bench/                    # M-competition benchmark harness
tests/                     # pytest suite
```

The engine is a plain Python library with a thin CLI wrapper — nothing in
it assumes a particular calling convention, so it stays cleanly usable from
scripts, notebooks, services, or (later) an MCP server without rework.

## Roadmap (not in this repo yet)

This repo is Phase 1: the core forecasting engine. Deliberately **not**
included, planned for later:

- **Override & audit trail.** Planner overrides on top of the statistical
  forecast, with a full audit history — the accountability layer that
  actually differentiates enterprise FP&A tooling.
- **Hierarchical reconciliation.** Rolling SKU-level forecasts up through
  product/region/company hierarchies with coherent totals.
- **Forecast Value Added (FVA) scoring.** Quantifying whether a planner's
  override actually improved accuracy versus the statistical baseline.
- **An MCP server for natural-language access.** Not a hosted chatbot or a
  bundled UI — a thin MCP layer over this same engine, so any MCP-aware
  client can drive it conversationally.

## License

Apache-2.0. See [LICENSE](LICENSE).
