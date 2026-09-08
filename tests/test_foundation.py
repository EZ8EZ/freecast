from __future__ import annotations

import datetime

import polars as pl
import pytest

torch = pytest.importorskip("torch")
t0_pkg = pytest.importorskip("t0")

from freecast.foundation import T0Model, levels_supported  # noqa: E402


def _tiny_forecaster():
    return t0_pkg.T0Forecaster(
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=8,
        group_every_n=1,
        dropout=0.0,
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
    ).eval()


@pytest.fixture
def tiny_t0_model() -> T0Model:
    return T0Model(forecaster=_tiny_forecaster())


def _series_df(n: int = 64) -> pl.DataFrame:
    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "unique_id": ["a"] * n + ["b"] * n,
            "ds": dates + dates,
            "y": [float(i) for i in range(n)] + [float(2 * i) for i in range(n)],
        }
    )


def test_levels_supported():
    assert levels_supported((80,))
    assert levels_supported((50, 80))
    assert not levels_supported((95,))
    assert not levels_supported((80, 95))


def test_predict_rejects_unsupported_level(tiny_t0_model):
    with pytest.raises(ValueError, match="only supports"):
        tiny_t0_model.predict(_series_df(), h=5, freq="1d", levels=(95,))


def test_predict_shape_and_columns(tiny_t0_model):
    df = _series_df()
    out = tiny_t0_model.predict(df, h=5, freq="1d", levels=(80,))

    assert set(out.columns) == {"unique_id", "ds", "T0", "T0-lo-80", "T0-hi-80"}
    assert out.height == 2 * 5
    for uid in ("a", "b"):
        sub = out.filter(pl.col("unique_id") == uid).sort("ds")
        assert sub.height == 5
        assert sub["ds"].min() > df.filter(pl.col("unique_id") == uid)["ds"].max()
        assert (sub["T0-lo-80"] <= sub["T0-hi-80"]).all()
