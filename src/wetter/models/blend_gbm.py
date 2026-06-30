from __future__ import annotations
import lightgbm as lgb
import numpy as np
import polars as pl

_CANDIDATES = [
    "t_icon_d2", "t_icon_eu", "t_icon_global", "t_gfs_seamless", "t_ecmwf_ifs025",
    "t_mean", "t_median", "t_spread", "t_min", "t_max",
    # auxiliary weather predictors (cross-model means)
    "rh_mean", "cloud_mean", "wind_mean", "pmsl_mean", "rad_mean",
    # interactions
    "clearnight_cool", "cloud_x_wind", "cloud_x_hourcos", "rh_night",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_time_h",
    "t_obs_at_issue", "t_clim", "station_vs_grid_elev_diff",
    "recent_bias_icon_d2", "recent_bias_icon_eu", "recent_bias_icon_global",
    "recent_bias_gfs_seamless", "recent_bias_ecmwf_ifs025",
]

_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)

# compact chronological-CV search grid for tune_hyperparams
_GRID = [
    {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
    for nl in (31, 63)
    for lr in (0.03, 0.05)
    for mcs in (20, 50)
]


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in _CANDIDATES if c in df.columns]


def _X(df: pl.DataFrame, features: list[str]):
    return df.select(features).to_pandas()


def train_point(
    train: pl.DataFrame, features: list[str], params: dict | None = None
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression_l1", **(params or _PARAMS))
    model.fit(_X(train, features), train["t_obs"].to_numpy())
    return model


def predict(model: lgb.LGBMRegressor, df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(_X(df, features))


def train_quantiles(
    train: pl.DataFrame, features: list[str], quantiles=(0.1, 0.5, 0.9),
    params: dict | None = None,
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    y = train["t_obs"].to_numpy()
    X = _X(train, features)
    for q in quantiles:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **(params or _PARAMS))
        m.fit(X, y)
        models[q] = m
    return models


def predict_quantiles(models, df, features) -> dict[float, np.ndarray]:
    X = _X(df, features)
    return {q: m.predict(X) for q, m in models.items()}


def tune_hyperparams(
    train: pl.DataFrame, valid: pl.DataFrame, features: list[str]
) -> dict:
    """Pick GBM params minimising point MAE on a chronological validation fold."""
    yv = valid["t_obs"].to_numpy()
    best, best_mae = dict(_PARAMS), float("inf")
    for g in _GRID:
        params = {**_PARAMS, **g}
        model = train_point(train, features, params)
        mae = float(np.mean(np.abs(predict(model, valid, features) - yv)))
        if mae < best_mae:
            best, best_mae = params, mae
    return best
