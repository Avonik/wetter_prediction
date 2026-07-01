import numpy as np
import polars as pl
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
