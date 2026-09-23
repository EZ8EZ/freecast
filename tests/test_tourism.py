"""Offline tests for the Tourism competition loader and paper-exact metrics.

The real-data check (the paper's own Naive/SNaive rows reproduce to the
digit) needs the dataset download and runs via ``freecast bench tourism``.
These tests pin the parsing, dating, and metric definitions it relies on.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import numpy as np
import polars as pl
import pytest

from freecast.bench import tourism
from freecast.bench.published_results import TOURISM


def _zip_bytes() -> bytes:
    # Two quarterly series in the competition's wide layout: name, length,
    # start year, start quarter, then values (short columns padded blank).
    q_in = "q1,q2\n4,3\n2000,2001\n1,3\n10,5\n20,6\n30,7\n40,\n"
    # q2's hold-out header deliberately repeats its last training period,
    # like the real Y18: the loader must date it as a continuation anyway.
    q_oos = "q1,q2\n2,2\n2001,2002\n1,1\n50,8\n60,9\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("quarterly_in.csv", q_in)
        zf.writestr("quarterly_oos.csv", q_oos)
    return buf.getvalue()


@pytest.fixture
def fake_zip(monkeypatch):
    monkeypatch.setattr(tourism, "_fetch_zip", lambda data_dir: _zip_bytes())


def test_load_parses_wide_layout_and_dates_holdout_as_continuation(fake_zip, tmp_path):
    train, test = tourism.load_tourism("Quarterly", tmp_path)

    q1 = train.filter(pl.col("unique_id") == "q1").sort("ds")
    assert q1["y"].to_list() == [10.0, 20.0, 30.0, 40.0]
    assert q1["ds"].to_list() == [
        dt.date(2000, 1, 1),
        dt.date(2000, 4, 1),
        dt.date(2000, 7, 1),
        dt.date(2000, 10, 1),
    ]
    q2_train = train.filter(pl.col("unique_id") == "q2").sort("ds")
    assert q2_train["ds"].to_list()[0] == dt.date(2001, 7, 1)
    assert q2_train["ds"].to_list()[-1] == dt.date(2002, 1, 1)

    q2_test = test.filter(pl.col("unique_id") == "q2").sort("ds")
    assert q2_test["ds"].to_list() == [dt.date(2002, 4, 1), dt.date(2002, 7, 1)]
    assert q2_test["y"].to_list() == [8.0, 9.0]


def test_unknown_group_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown Tourism group"):
        tourism.load_tourism("Weekly", tmp_path)


def test_fetch_rejects_content_with_wrong_hash(monkeypatch, tmp_path):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tourism.urllib.request, "urlopen", lambda url, timeout: _Resp(b"<html>"))
    with pytest.raises(RuntimeError, match="unexpected content"):
        tourism._fetch_zip(tmp_path)
    assert not (tmp_path / "tourism" / tourism.ZIP_NAME).exists()


def test_paper_mape_is_plain_mape_averaged_per_series():
    joined = pl.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "y": [100.0, 200.0, 10.0, 10.0],
            "y_hat": [110.0, 180.0, 10.0, 15.0],
        }
    )
    # a: (10% + 10%) / 2 = 10%; b: (0% + 50%) / 2 = 25%; mean = 17.5%
    assert tourism.paper_mape(joined) == pytest.approx(17.5)


def test_paper_mase_scales_over_training_and_holdout():
    train = pl.DataFrame({"unique_id": ["a"] * 3, "ds": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    test = pl.DataFrame({"unique_id": ["a"] * 2, "ds": [4, 5], "y": [7.0, 11.0]})
    joined = test.with_columns(pl.Series("y_hat", [5.0, 9.0]))
    # Full-series |diff| = 1, 1, 4, 4 -> scale 2.5; MAE = 2 -> MASE 0.8.
    # (A training-only scale of 1.0 would give 2.0.)
    assert tourism.paper_mase(joined, train, test, m=1) == pytest.approx(0.8)


def test_naive_forecast_repeats_last_season():
    train = pl.DataFrame({"unique_id": ["a"] * 6, "ds": list(range(6)), "y": np.arange(6.0)})
    test = pl.DataFrame({"unique_id": ["a"] * 5, "ds": list(range(6, 11)), "y": np.zeros(5)})
    seasonal = tourism.naive_forecast(train, test, m=3).sort("ds")
    assert seasonal["y_hat"].to_list() == [3.0, 4.0, 5.0, 3.0, 4.0]
    plain = tourism.naive_forecast(train, test, m=1)
    assert set(plain["y_hat"].to_list()) == {5.0}


def test_published_tourism_table_includes_forecast_pro_for_every_group():
    assert set(TOURISM) == set(tourism.TOURISM_GROUPS)
    for methods in TOURISM.values():
        assert "ForePro" in methods
