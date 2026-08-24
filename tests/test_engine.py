from __future__ import annotations

from freecast.engine import ForecastEngine, infer_season_length


def test_infer_season_length():
    assert infer_season_length("MS") == 12
    assert infer_season_length("1mo") == 1  # unknown key falls back to non-seasonal
    assert infer_season_length("D") == 7
    assert infer_season_length("Q") == 4
    assert infer_season_length(1) == 1


def test_engine_end_to_end_mixed(mixed_series_df):
    engine = ForecastEngine(h=6, freq="1mo", n_windows=1)
    result = engine.run(mixed_series_df)

    all_ids = set(mixed_series_df["unique_id"].unique().to_list())
    assert set(result.selection["unique_id"].to_list()) == all_ids
    assert set(result.classification["unique_id"].to_list()) == all_ids
    assert set(result.forecasts["unique_id"].unique().to_list()) == all_ids

    for uid in all_ids:
        n_rows = result.forecasts.filter(result.forecasts["unique_id"] == uid).height
        assert n_rows == 6

    assert "lo-80" in result.forecasts.columns
    assert "hi-80" in result.forecasts.columns
    assert "lo-95" in result.forecasts.columns
    assert "hi-95" in result.forecasts.columns
    lo_le_hi = (result.forecasts["lo-80"] <= result.forecasts["hi-80"]).all()
    assert lo_le_hi


def test_engine_regular_only(regular_series_df):
    engine = ForecastEngine(h=6, freq="1mo", n_windows=1)
    result = engine.run(regular_series_df)
    assert (result.classification["category"] == "smooth").all()
    assert result.forecasts.height == 4 * 6
