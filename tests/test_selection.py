from __future__ import annotations

from freecast.selection import default_intermittent_models, default_regular_models, select_models


def test_select_models_regular(regular_series_df):
    models = default_regular_models(season_length=12)
    result = select_models(
        regular_series_df, h=6, freq="1mo", season_length=12, models=models, n_windows=1
    )
    assert set(result.best_model["unique_id"].to_list()) == {
        "series_0",
        "series_1",
        "series_2",
        "series_3",
    }
    valid_names = {getattr(m, "alias", type(m).__name__) for m in models}
    assert set(result.best_model["model"].to_list()) <= valid_names
    assert (result.best_model["mase"] >= 0).all()


def test_select_models_intermittent(intermittent_series_df):
    models = default_intermittent_models()
    result = select_models(
        intermittent_series_df, h=6, freq="1mo", season_length=12, models=models, n_windows=1
    )
    valid_names = {getattr(m, "alias", type(m).__name__) for m in models}
    assert set(result.best_model["model"].to_list()) <= valid_names


def test_select_models_unknown_metric_raises(regular_series_df):
    import pytest

    with pytest.raises(ValueError, match="Unknown metric"):
        select_models(regular_series_df, h=6, freq="1mo", season_length=12, metric="nope")
