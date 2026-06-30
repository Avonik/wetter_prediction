import polars as pl
from datetime import datetime, timezone
from wetter.split import chronological_split


def test_split_partitions_chronologically():
    times = [datetime(2024, m, 1, tzinfo=timezone.utc) for m in range(1, 13)]
    df = pl.DataFrame({"valid_time": times, "x": list(range(12))})
    tr, ca, te = chronological_split(df, train_end="2024-07-01", cal_end="2024-10-01")
    assert tr.height == 6 and ca.height == 3 and te.height == 3
    assert tr["valid_time"].max() < ca["valid_time"].min() < te["valid_time"].min()
