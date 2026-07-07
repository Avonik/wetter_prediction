import httpx
import polars as pl
import pytest
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


def test_cached_parquet_many_runs_and_caches(tmp_path):
    built = []  # list.append is atomic under the GIL — safe across threads

    def make_builder(k):
        def builder():
            built.append(k)
            return pl.DataFrame({"k": [k]})

        return builder

    items = [(tmp_path / f"{k}.parquet", make_builder(k)) for k in range(6)]
    out = io.cached_parquet_many(items, concurrency=4)
    assert [d["k"][0] for d in out] == list(range(6))  # order preserved
    assert sorted(built) == list(range(6))  # each built exactly once
    # second call: all cached -> no new builder calls
    out2 = io.cached_parquet_many(items, concurrency=4)
    assert len(built) == 6
    assert [d["k"][0] for d in out2] == list(range(6))


def test_get_json_raises_transient_on_persistent_429(monkeypatch):
    """A run of 429s must raise Transient (a distinct type), NOT httpx.HTTPStatusError —
    so callers don't mistake rate-limiting for a genuine 'resource missing' and cache it."""

    class Resp:
        status_code = 429

        def json(self):
            return {"reason": "Too many concurrent requests"}

        def raise_for_status(self):  # never reached on the 429 path
            raise AssertionError("should not be called for 429")

    monkeypatch.setattr(io.time, "sleep", lambda *_: None)  # no real backoff waits
    monkeypatch.setattr(io._CLIENT, "get", lambda *a, **k: Resp())
    with pytest.raises(io.Transient):
        io.get_json("http://x", {})
    assert not issubclass(io.Transient, httpx.HTTPError)  # must be distinguishable


def test_cached_parquet_many_drops_failing_builder(tmp_path):
    """A transiently-failing builder is dropped (not written empty), so the next run
    retries it — the self-healing property that prevents poisoning the cache."""

    def good():
        return pl.DataFrame({"k": [1]})

    def bad():
        raise io.Transient("throttled")

    pg, pb = tmp_path / "good.parquet", tmp_path / "bad.parquet"
    out = io.cached_parquet_many([(pb, bad), (pg, good)], concurrency=2)
    assert len(out) == 1 and out[0]["k"][0] == 1  # failing job omitted from results
    assert pg.exists()  # success cached
    assert not pb.exists()  # failure NOT cached -> retried next run


def test_cached_parquet_force_rebuilds(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return pl.DataFrame({"a": [calls["n"]]})

    p = tmp_path / "x.parquet"
    io.cached_parquet(p, builder)
    io.cached_parquet(p, builder, force=True)
    assert calls["n"] == 2
