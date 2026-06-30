import numpy as np
import polars as pl
from wetter.models import blend_gbm as G


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t_icon = rng.normal(10, 5, n)
    noise = rng.normal(0, 1, n)
    # truth = icon minus its +2 bias plus mild noise
    t_obs = t_icon - 2.0 + noise
    return pl.DataFrame(
        {
            "t_icon_d2": t_icon, "t_mean": t_icon, "t_median": t_icon,
            "t_spread": np.zeros(n), "t_min": t_icon, "t_max": t_icon,
            "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
            "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
            "lead_time_h": np.full(n, 24), "t_obs_at_issue": t_obs,
            "t_clim": np.full(n, 10.0), "t_obs": t_obs,
        }
    )


def test_point_blend_beats_raw_on_synth():
    df = _synth()
    tr, te = df.head(300), df.tail(100)
    feats = G.feature_columns(tr)
    model = G.train_point(tr, feats)
    pred = G.predict(model, te, feats)
    y = te["t_obs"].to_numpy()
    raw_mae = np.mean(np.abs(te["t_icon_d2"].to_numpy() - y))
    blend_mae = np.mean(np.abs(pred - y))
    assert blend_mae < raw_mae  # learned away the +2 bias


def test_tune_returns_params_no_worse_than_default():
    df = _synth(600)
    tr, va = df.head(450), df.tail(150)
    feats = G.feature_columns(tr)
    params = G.tune_hyperparams(tr, va, feats)
    assert {"num_leaves", "learning_rate", "min_child_samples"} <= set(params)
    yv = va["t_obs"].to_numpy()
    tuned_mae = np.mean(np.abs(G.predict(G.train_point(tr, feats, params), va, feats) - yv))
    default_mae = np.mean(np.abs(G.predict(G.train_point(tr, feats), va, feats) - yv))
    # the default config is inside the search grid, so the tuned pick can't be worse
    assert tuned_mae <= default_mae + 1e-9


def test_quantiles_are_ordered_on_average():
    df = _synth()
    feats = G.feature_columns(df)
    models = G.train_quantiles(df.head(300), feats)
    preds = G.predict_quantiles(models, df.tail(100), feats)
    assert np.mean(preds[0.1]) < np.mean(preds[0.5]) < np.mean(preds[0.9])
