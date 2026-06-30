import polars as pl
from wetter.data import io


def test_month_chunks_spans_year_boundary():
    chunks = io.month_chunks("2023-12-01", "2024-02-15")
    assert chunks == [
        ("2023-12-01", "2023-12-31"),
        ("2024-01-01", "2024-01-31"),
        ("2024-02-01", "2024-02-15"),
    ]


def test_cached_parquet_writes_then_reads(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return pl.DataFrame({"a": [1, 2]})

    p = tmp_path / "x.parquet"
    df1 = io.cached_parquet(p, builder)
    df2 = io.cached_parquet(p, builder)  # second call hits cache
    assert calls["n"] == 1
    assert df1.equals(df2)


def test_cached_parquet_force_rebuilds(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return pl.DataFrame({"a": [calls["n"]]})

    p = tmp_path / "x.parquet"
    io.cached_parquet(p, builder)
    io.cached_parquet(p, builder, force=True)
    assert calls["n"] == 2
