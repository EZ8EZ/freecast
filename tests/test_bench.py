"""Import-path and freq-consistency regression tests for the bench package.

Nothing here downloads M3 (that's an opt-in, network-dependent smoke test
run manually via `freecast bench m3`) — this only guards the two bugs the
package previously shipped with:

1. ``bench`` lived outside ``src/freecast/``, so it was importable when
   running from a repo checkout (pytest, ``python -m bench.runner``) but
   *not* for anyone who actually ``pip install``-ed the package — the
   installed ``freecast`` console script raised ``ModuleNotFoundError:
   No module named 'bench'`` unconditionally. Nothing in the test suite
   imported it, so this was invisible until exercised as an installed CLI.
2. ``M3_GROUPS`` fed Polars-style freq strings ("1mo", "1q", ...) into
   ``ForecastEngine``, whose season-length inference only recognized
   pandas-style keys — every M3 group silently ran with season_length=1.
"""

from __future__ import annotations

from freecast.bench.published_results import M3_OVERALL
from freecast.bench.runner import M3_GROUPS, BenchSummary
from freecast.freq import resolve_freq


def test_bench_importable_as_freecast_subpackage():
    # Regression guard for the ModuleNotFoundError described above: this
    # must resolve via the *installed* package path (freecast.bench), not
    # rely on pytest's rootdir-insertion import magic finding a top-level
    # `bench` directory.
    assert M3_GROUPS
    assert BenchSummary is not None


def test_m3_overall_reference_table_shape():
    assert "ForecastPro" in M3_OVERALL
    for name, (smape, mase) in M3_OVERALL.items():
        assert isinstance(name, str)
        assert smape > 0
        assert mase > 0


def test_m3_group_season_lengths_reach_the_engine():
    # M3_GROUPS' third tuple element (h, freq, season_length) is what
    # run_m3_group passes to ForecastEngine as an explicit season_length
    # override — it must NOT be silently re-derived from freq instead. The
    # "Other" group is the case that makes this matter: datasetsforecast
    # stands in fake daily dates for it (real M3 "Other" series have no
    # calendar meaning at all), so resolve_freq("1d") infers season_length=7
    # (weekly) purely from the placeholder date step, which would inject
    # seasonality the data doesn't have. M3_GROUPS correctly declares 1 for
    # it; ForecastEngine must honor that override rather than defaulting.
    from freecast.engine import ForecastEngine

    for group, (h, freq, expected_season_length) in M3_GROUPS.items():
        engine = ForecastEngine(h=h, freq=freq, season_length=expected_season_length)
        assert engine.season_length == expected_season_length, (
            f"M3 group {group!r}: engine.season_length="
            f"{engine.season_length}, but M3_GROUPS declares {expected_season_length}"
        )

    # And the groups with real calendar periodicity should also match what
    # freq inference would have picked anyway (Yearly, Quarterly, Monthly) —
    # only "Other"'s fabricated daily dates diverge from their true intent.
    for group in ("Yearly", "Quarterly", "Monthly"):
        _h, freq, expected_season_length = M3_GROUPS[group]
        assert resolve_freq(freq).season_length == expected_season_length
