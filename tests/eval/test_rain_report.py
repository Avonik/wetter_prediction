from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from wetter.eval import rain_report


def test_rain_backtest_compares_only_future_holdout():
    n = 4000
    rng = np.random.default_rng(4)
    precip = np.clip(rng.exponential(0.3, n) - 0.1, 0.0, None)
    observed = np.clip(precip + rng.normal(0, 0.12, n), 0.0, None)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "valid_time": [start + timedelta(hours=i) for i in range(n)],
            "p_obs": observed,
            "precip_mean": precip,
            "precip_max": precip * 1.5,
            "precip_prob": (precip >= 0.1).astype(float),
            "cloud_mean": rng.uniform(0, 100, n),
            "rh_mean": rng.uniform(40, 100, n),
        }
    ).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))

    result = rain_report.evaluate_options(
        frame, train_end="2026-03-01", cal_end="2026-04-01"
    )

    assert result["rows"]["train"] == 59 * 24
    assert result["rows"]["calibration"] == 31 * 24
    assert set(result["options"]) == {"raw", "isotonic", "beta"}
    for metrics in result["options"].values():
        assert 0.0 <= metrics["brier"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
    assert "beta" in rain_report.format_options(result)
