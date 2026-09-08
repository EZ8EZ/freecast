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

Published M3 competition results (overall average across all 3,003 series):

| Method | sMAPE | MASE |
|---|---|---|
| Theta | 13.01 | 1.39 |
| **ForecastPro** (commercial, named M3 entrant) | 13.19 | 1.47 |
| ForecastX | 13.49 | 1.42 |
| ETS | 13.07 | 1.43 |
| AutoARIMA | 13.57 | 1.45 |

freecast, run end-to-end (validate → classify → CV-select → conformal
forecast) against the real M3 data, one dataset download, ~25.8m total on a
single laptop-class VM, zero series dropped:

| Group | n | sMAPE | MASE | Time |
|---|---|---|---|---|
| Yearly | 645 | 18.02 | 3.04 | 72.3s |
| Quarterly | 756 | 9.72 | 1.15 | 225.3s |
| Monthly | 1,428 | 14.50 | 0.85 | 1,199.9s |
| Other | 174 | 4.52 | 1.91 | 52.5s |
| **Overall (weighted)** | **3,003** | **13.47** | **1.46** | **~25.8m** |

freecast's overall sMAPE (13.47) and MASE (1.46) land in the same band as
Theta, ForecastPro, ETS, and AutoARIMA above — competitive with, though not
quite beating, the best M3 entrants, from a from-scratch CV-based
model-selection pipeline with zero per-series tuning. (Yearly is the
hardest M3 category for every entrant, ours included — short series and a
6-step horizon leave little room for any method to do well; per-group
numbers above are directly reproducible with `freecast bench m3 --group
Yearly`.)

**A correctness note, in the interest of the "verify it yourself" pitch
actually meaning something:** an earlier internal run of this same harness
reported Monthly at sMAPE 15.32 / MASE 0.95 and a ~6m19s total. Both were
artifacts of a bug where the season length (12 for monthly data) was never
actually reaching the model pool — every M3 group was silently fit with no
seasonality at all, and the run was faster only because non-seasonal model
fits are cheaper. Fixing it (`freecast.freq.resolve_freq`, plus an explicit
`season_length` override in `ForecastEngine` for cases — like M3's "Other"
group, whose dates are synthetic — where the periodicity implied by a
timestamp's frequency doesn't reflect anything real in the data) changed
Monthly meaningfully (a genuine ~10% MASE improvement once seasonal models
are actually seasonal) and left Yearly, Quarterly, and Other essentially
unchanged, which is exactly what should happen: Yearly has no sub-annual
periodicity to find either way, Quarterly's models evidently found the
4-period seasonality made no difference to their own AIC-selected
specification, and Other's dates are fabricated (no real calendar meaning),
so a spurious weekly period from the fix's own default inference had to be
overridden back to 1 rather than trusted. The current numbers above are
this corrected run.

Run `freecast bench m3` yourself to reproduce these numbers, or break them
down by frequency group with `freecast bench m3 --group Monthly`. Raw
per-group output:
[`src/freecast/bench/results/m3_summary.json`](src/freecast/bench/results/m3_summary.json).

M4 and Tourism harnesses are stubbed out in `src/freecast/bench/` for a
follow-up; M5 is out of scope for now given its size.

## Optional: zero-shot foundation model candidate

freecast can optionally include [t0](https://github.com/theforecastingcompany/tfc-t0)
— The Forecasting Company's ~102M-parameter, Apache-2.0, zero-shot
foundation model — as an extra candidate in the regular-series pool,
competing on the same backtested accuracy metric as AutoETS/AutoARIMA/
AutoTheta/AutoCES. Since it needs no per-series training, it's a natural
fit for series with too little history to backtest a statistical model
against reliably.

```bash
pip install "freecast[foundation]"
```

t0's weights live in a gated Hugging Face repo: request access on the
[model page](https://huggingface.co/theforecastingcompany/t0-alpha), then
run `hf auth login`. freecast never bundles or auto-downloads weights.

```python
engine = ForecastEngine(h=12, freq="MS", use_foundation_model=True)
```

or `freecast run ... --use-foundation-model`. t0 only ever emits five fixed
quantiles (0.1/0.25/0.5/0.75/0.9), so it can only serve prediction-interval
levels derivable from that set — 80 and/or 50; requesting any other level
makes it ineligible as a candidate for that run rather than erroring.
If the extra isn't installed, or Hugging Face access isn't set up, freecast
degrades gracefully to the statistical pool alone — this is exercised in
`tests/test_engine.py`.

## Overrides, audit trail, and Forecast Value Added

The statistical forecast is a starting point, not the final answer — planners
override it, and someone eventually needs to answer "who changed this number,
when, and why," and whether that override actually helped.

**`OverrideStore`** is an append-only audit trail backed by a local DuckDB
file. Every override is recorded, never mutated — superseding a prior
override for the same series/date keeps the old row intact with a pointer to
what replaced it, so the full history is always reconstructable.

```python
from freecast.overrides import OverrideStore

with OverrideStore("overrides.duckdb") as store:
    store.add(
        unique_id="SKU123", ds=date(2024, 3, 1), model="AutoETS",
        baseline_y_hat=142.0, override_y_hat=180.0,
        author="j.smith", reason="regional promo confirmed",
    )
    final = store.apply(result.forecasts)  # adds y_hat_final, is_overridden
```

or from the CLI: `freecast override add --unique-id SKU123 --ds 2024-03-01
--model AutoETS --baseline 142.0 --value 180.0 --author j.smith`, and
`freecast override log` to see the full trail.

**Forecast Value Added (FVA)** answers the follow-up question: did that
override — or the statistical model itself — actually beat a naive baseline?
Published research finds 40-60% of manual overrides make forecasts *worse*;
FVA is how you find out which ones, per the standard practice described in
Gilliland's *Business Forecasting*.

```python
from freecast.fva import compute_fva

result = compute_fva(train_df, forecasts, actuals, freq="MS", overrides=overrides_df)
result.per_series   # unique_id, naive/statistical/override error, and *_fva columns
result.overall      # same columns, averaged
```

or `freecast fva --train train.csv --forecasts forecasts.parquet --actuals
actuals.csv --freq MS --overrides-db overrides.duckdb`. Positive FVA means
that layer beat its baseline; negative means it made the forecast worse —
the standard evidence base for retiring an override layer, or a planner's
adjustments, that isn't earning its keep.

## Architecture

```
src/freecast/
├── freq.py          # canonical pandas-/Polars-style frequency handling
├── contract.py      # data validation — fails loudly on bad input
├── intermittent.py  # ADI/CV² intermittent-demand classification
├── selection.py     # cross-validation-driven model selection
├── intervals.py     # conformal prediction interval builder
├── foundation.py    # optional t0 zero-shot foundation-model candidate
├── overrides.py     # planner override log with a durable audit trail
├── fva.py           # Forecast Value Added scoring
├── engine.py        # orchestrates the forecasting pipeline
├── cli.py           # the `freecast` command
└── bench/           # M-competition benchmark harness
tests/                # pytest suite
```

The engine is a plain Python library with a thin CLI wrapper — nothing in
it assumes a particular calling convention, so it stays cleanly usable from
scripts, notebooks, services, or (later) an MCP server without rework.

## Roadmap (not in this repo yet)

Phase 1 (the core engine) and the override/audit-trail/FVA workflow layer are
here. Deliberately **not** included yet:

- **Hierarchical reconciliation.** Rolling SKU-level forecasts up through
  product/region/company hierarchies with coherent totals.
- **An MCP server for natural-language access.** Not a hosted chatbot or a
  bundled UI — a thin MCP layer over this same engine, so any MCP-aware
  client can drive it conversationally.

## License

Apache-2.0. See [LICENSE](LICENSE).
