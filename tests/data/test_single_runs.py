import httpx
import polars as pl
from wetter.data import single_runs as sr

PAYLOAD = {
    "elevation": 22.0,
    "hourly": {
        "time": ["2026-06-25T00:00", "2026-06-25T01:00", "2026-06-25T02:00"],
        "temperature_2m": [20.0, 19.5, 19.0],
        "cloud_cover": [10.0, 20.0, None],
    },
}


def test_parse_run_computes_lead_from_run_time():
    df = sr.parse_run(PAYLOAD, "icon_d2", "2026-06-25T00:00")
    assert set(df.columns) == set(sr._SCHEMA.keys())
    # temperature lead 0 is dropped (we keep >=1 only in fetch, but parse keeps all leads)
    t = df.filter(pl.col("variable") == "t").sort("lead_time_h")
    assert t["lead_time_h"].to_list() == [0, 1, 2]
    assert t["value"].to_list() == [20.0, 19.5, 19.0]
    assert df["run_time"].dtype == pl.Datetime("us", "UTC")
    # cloud has a null at lead 2 -> dropped
    assert df.filter(pl.col("variable") == "cloud").height == 2


def test_fetch_runs_handles_missing_run_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get_json(url, params):
        calls["n"] += 1
        if params["run"].startswith("2026-06-25"):
            return PAYLOAD
        # simulate an unavailable run
        raise httpx.HTTPStatusError("400", request=None, response=None)

    monkeypatch.setattr(sr.io, "get_json", fake_get_json)
    df = sr.fetch_runs(
        "2026-06-25", "2026-06-26", models=["icon_d2"], run_hours=(0,), cache_dir=tmp_path
    )
    # only 2026-06-25 returned data (leads 1,2 for t and 1 for cloud); 06-26 was a 400 -> empty
    assert df.filter(pl.col("variable") == "t")["lead_time_h"].to_list() == [1, 2]
    n1 = calls["n"]
    sr.fetch_runs(
        "2026-06-25", "2026-06-26", models=["icon_d2"], run_hours=(0,), cache_dir=tmp_path
    )
    assert calls["n"] == n1  # both runs cached (including the empty 400)
