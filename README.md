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
- **Headless by design.** A library, a thin CLI, and an MCP server on top of
  it. No GUI, no bundled dashboard — this is meant to be called from your own
  pipeline, or conversationally from any MCP-aware client.

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

### Exogenous regressors

Any column besides `unique_id`, `ds`, `y` is an exogenous regressor — a
promo flag, price, a known-holiday indicator. Forecasting past the end of
history needs *known future values* for those (freecast can't invent your
promo calendar), so pass an `X_df` with the same regressor columns covering
exactly the next `h` periods per series:

```python
result = engine.run(df, X_df=future_regressors)  # unique_id, ds, promo, ...
```

or `freecast run sales.csv --regressors-path future_promo.csv ...`. Omitting
`X_df` when the input has regressor columns is a clear, specific error
rather than a silent drop — freecast used to crash deep inside statsforecast
on this; now it tells you exactly what's missing before it gets there.

Only AutoARIMA in the regular-series pool actually has a mechanism for
exogenous regressors — AutoETS/AutoTheta/AutoCES accept the extra columns
without erroring but silently ignore them (classical exponential smoothing
and Theta have no covariate concept). This isn't a gap freecast codes
around: it's the CV-based selection the whole engine is built on working
correctly. If a regressor is genuinely predictive, AutoARIMA backtests
better — because it alone can use that information — and wins selection on
its own merits, no special-casing required.

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

## Exception reporting

At thousands-of-SKU scale nobody reviews every forecast. Forecast Pro sells
"exception reporting" as a named feature for exactly this reason; `freecast`
surfaces the same idea as a data query, not a dashboard — a handful of
independently-thresholded, independently-interpretable flags per series, so
a planner challenged on "why is this flagged" gets a specific answer, not
"the model said so."

```python
from freecast.exceptions import find_exceptions

report = find_exceptions(train_df, result.forecasts, result.selection)
report.flagged   # unique_id, flags, n_flags, plus the diagnostic columns each flag used
report.summary   # {"poor_accuracy": 12, "wide_interval": 4, "large_jump": 7}
```

Four flags, each opt-out via its own threshold: `poor_accuracy` (CV MASE/RMSSE
above 1.0 — no better than naive), `wide_interval` (prediction interval wider
than half the point forecast), `large_jump` (forecast mean differs from the
last actual by more than 50%), and `demand_type_changed` (optional — pass a
prior run's demand classification to flag series that crossed the
regular/intermittent boundary). Results are ranked by how many flags fired.

or `freecast exceptions --train sales.csv --output-dir freecast_output`
against a prior `freecast run`'s output, or `freecast_get_exceptions` from
the MCP server.

## Hierarchical reconciliation

Planners don't forecast one SKU in isolation — a region or category total
needs to equal the sum of its SKU-level forecasts, or the number is useless
for S&OP and finance review. Forecasting every level independently produces
*incoherent* totals; `freecast.hierarchy` fixes that by forecasting every
level with the same engine (each level is just another synthetic series to
it) and reconciling with [`hierarchicalforecast`](https://github.com/Nixtla/hierarchicalforecast)
(Apache-2.0), defaulting to MinTrace with structural scaling — a standard,
well-behaved method that needs only actuals, not in-sample residuals.

```python
from freecast.hierarchy import reconcile

result = reconcile(df, spec=[["category"], ["category", "region"]], h=12, freq="MS")
result.forecasts   # unique_id, ds, y_hat, y_hat/BottomUp, y_hat/MinTrace_method-wls_struct
```

or `freecast reconcile sales.csv --hierarchy category,region --horizon 12
--freq MS` — every prefix of `--hierarchy`'s columns becomes a level, bottom
level included automatically. `freecast.hierarchy.check_coherence()` verifies
the property reconciliation exists to guarantee: every aggregate equals the
sum of its children, which the unreconciled per-level forecast generally
does not.

## MCP server: natural-language access

`freecast` ships an [MCP](https://modelcontextprotocol.io) server so any
MCP-aware client (Claude Code, Claude Desktop, etc.) can drive the whole
workflow conversationally, pointed at your local project files — no hosted
chatbot, no bundled UI, just a thin layer over the same library everything
else in this README uses:

```bash
freecast mcp
```

Seven tools cover the full loop: `freecast_run_forecast` / `freecast_get_forecast`
(run the engine, read results back without re-running), `freecast_add_override`
/ `freecast_list_overrides` (the audit trail), `freecast_get_fva_report`
(did that override actually help?), `freecast_reconcile_hierarchy`, and
`freecast_get_exceptions` (which series need a planner's review). Each
one is a direct wrapper around the corresponding library call — no new
forecasting logic lives in `mcp_server.py` — and large results are written to
disk with a capped, filterable inline preview rather than dumped whole into
context.

Point your MCP client's config at `freecast mcp` (stdio transport) and ask
things like *"forecast next quarter for sales.csv at monthly granularity"*,
*"override SKU123's March forecast to 180, the regional promo is
confirmed"*, or *"did last month's overrides actually help accuracy?"*

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
├── hierarchy.py     # hierarchical reconciliation (coherent multi-level forecasts)
├── exceptions.py     # rule-based exception reporting for planner review
├── engine.py        # orchestrates the forecasting pipeline
├── mcp_server.py     # MCP server exposing the library to any MCP client
├── cli.py             # the `freecast` command
└── bench/               # M-competition benchmark harness
tests/                    # pytest suite
```

The engine is a plain Python library with a thin CLI wrapper and an MCP
server on top of it — nothing in the core assumes a particular calling
convention, so it stays cleanly usable from scripts, notebooks, services, or
conversationally without rework.

## Roadmap

The originally scoped feature set — the core engine, the override/audit-trail/
FVA workflow layer, hierarchical reconciliation, and the MCP server — is all
here. Ideas for what's next (not started): a hosted/multi-tenant deployment
option, richer collaborative consensus workflows across sales/ops/finance,
and M4/Tourism bench harnesses to match the M3 one.

## License

Apache-2.0. See [LICENSE](LICENSE).
