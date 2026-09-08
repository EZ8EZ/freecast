import pytest

from freecast.freq import resolve_freq


@pytest.mark.parametrize(
    ("freq", "expected_polars", "expected_pandas", "expected_season_length"),
    [
        ("MS", "1mo", "MS", 12),
        ("M", "1mo", "MS", 12),
        ("D", "1d", "D", 7),
        ("W", "1w", "W", 52),
        ("Q", "1q", "QS", 4),
        ("QS", "1q", "QS", 4),
        ("Y", "1y", "YS", 1),
        ("A", "1y", "YS", 1),
        ("H", "1h", "h", 24),
        # Already Polars-style — must resolve identically to the pandas alias
        # that means the same thing (this is the exact case that silently
        # broke season-length inference before resolve_freq existed).
        ("1mo", "1mo", "MS", 12),
        ("1d", "1d", "D", 7),
        ("1w", "1w", "W", 52),
        ("1q", "1q", "QS", 4),
        ("1y", "1y", "YS", 1),
    ],
)
def test_resolve_freq_dialects_agree(
    freq, expected_polars, expected_pandas, expected_season_length
):
    resolved = resolve_freq(freq)
    assert resolved.polars == expected_polars
    assert resolved.pandas == expected_pandas
    assert resolved.season_length == expected_season_length


def test_resolve_freq_multi_step_has_no_confident_season_length():
    resolved = resolve_freq("2MS")
    assert resolved.polars == "2mo"
    assert resolved.pandas == "2MS"
    assert resolved.season_length == 1


def test_resolve_freq_integer_passthrough():
    resolved = resolve_freq(3)
    assert resolved.polars == 3
    assert resolved.pandas == 3
    assert resolved.season_length == 1


def test_resolve_freq_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Unrecognized frequency"):
        resolve_freq("XYZ")


def test_resolve_freq_rejects_malformed_string():
    with pytest.raises(ValueError, match="Unrecognized frequency"):
        resolve_freq("!!!")
