import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta, timezone
from sklearn.metrics import roc_auc_score
from wetter.models import rain


def _synth(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pm = np.clip(rng.exponential(0.3, n) - 0.1, 0.0, None)  # precip_mean, mostly ~0
    p_obs = np.clip(pm * rng.uniform(0.5, 2.0, n) - 0.05 + rng.normal(0, 0.05, n), 0.0, None)
    return pl.DataFrame(
        {
            "precip_mean": pm, "precip_max": pm * 1.5, "precip_prob": (pm >= 0.1).astype(float),
            "cloud_mean": rng.uniform(0, 100, n), "rh_mean": rng.uniform(40, 100, n),
            "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
            "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
            "lead_time_h": np.full(n, 24),
            "p_obs": p_obs,
        }
    )


def test_exceedance_trains_and_separates():
    df = _synth()
    feats = rain.feature_columns(df)
    tr, te = df.head(2000), df.tail(1000)
    models = rain.train_exceedance(tr, feats)
    assert 0.1 in models
    probs = rain.predict_exceedance(models, te, feats)
    p = probs[0.1]
    assert p.min() >= 0.0 and p.max() <= 1.0
    y = (te["p_obs"] >= 0.1).to_numpy().astype(int)
    assert roc_auc_score(y, p) > 0.75  # separates rain from dry


def test_calibration_returns_valid_probs():
    df = _synth()
    feats = rain.feature_columns(df)
    tr, ca, te = df.head(1500), df.slice(1500, 700), df.tail(800)
    models = rain.train_exceedance(tr, feats)
    cals = rain.fit_calibration(models, ca, feats)
    cal = rain.apply_calibration(cals, rain.predict_exceedance(models, te, feats))
    assert cal[0.1].min() >= 0.0 and cal[0.1].max() <= 1.0


def test_beta_calibration_rejects_unknown_method():
    df = _synth()
    feats = rain.feature_columns(df)
    models = rain.train_exceedance(df.head(2000), feats)
    with pytest.raises(ValueError, match="Unknown rain calibration method"):
        rain.fit_calibration(models, df.tail(1000), feats, method="magic")


def test_train_rain_engine_uses_chronological_calibration_window():
    df = _synth(4000).with_columns(
        pl.datetime_range(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=3999),
            interval="1h",
            eager=True,
        ).alias("valid_time"),
        (pl.col("p_obs") >= 0.1).cast(pl.Int8).alias("rain_occ"),
    )

    artifact = rain.train_rain_engine(df, cal_window_days=30)

    assert artifact["calibration"] == "beta"
    assert 0.1 in artifact["cals"]
    assert artifact["fit_through"] < artifact["calibration_start"]
    assert artifact["trained_through"] == df["valid_time"].max()


def test_predict_rain_enforces_nested_thresholds(monkeypatch):
    monkeypatch.setattr(
        rain,
        "predict_exceedance",
        lambda models, df, feats: {
            0.1: np.array([0.4]),
            1.0: np.array([0.6]),
            5.0: np.array([0.5]),
        },
    )
    probabilities = rain.predict_rain(
        {"models": {}, "features": [], "cals": {}}, pl.DataFrame()
    )
    assert probabilities[0.1][0] == 0.4
    assert probabilities[1.0][0] == 0.4
    assert probabilities[5.0][0] == 0.4
