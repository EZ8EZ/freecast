from __future__ import annotations

from freecast.intermittent import classify_series, split_by_demand_type


def test_regular_series_classified_smooth(regular_series_df):
    result = classify_series(regular_series_df)
    assert (result["category"] == "smooth").all()


def test_sparse_series_classified_intermittent_family(intermittent_series_df):
    result = classify_series(intermittent_series_df)
    row = result.row(0, named=True)
    assert row["category"] in ("intermittent", "erratic", "lumpy")
    assert row["adi"] > 1


def test_all_zero_series_classified_intermittent():
    import polars as pl

    df = pl.DataFrame({"unique_id": ["z"] * 10, "ds": list(range(10)), "y": [0.0] * 10})
    result = classify_series(df)
    assert result.row(0, named=True)["category"] == "intermittent"


def test_split_by_demand_type(mixed_series_df):
    regular, sparse = split_by_demand_type(mixed_series_df)
    assert set(regular["unique_id"].unique().to_list()) == {
        "series_0",
        "series_1",
        "series_2",
        "series_3",
    }
    assert set(sparse["unique_id"].unique().to_list()) == {"series_sparse"}
