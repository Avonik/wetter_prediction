from __future__ import annotations
import math

import polars as pl

from wetter import config

_TAU = 2 * math.pi


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("valid_time").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("valid_time").dt.ordinal_day().cast(pl.Int32).alias("doy"),
        pl.col("valid_time").dt.month().cast(pl.Int32).alias("month"),
    ).with_columns(
        (pl.col("hour") * (_TAU / 24)).sin().alias("hour_sin"),
        (pl.col("hour") * (_TAU / 24)).cos().alias("hour_cos"),
        (pl.col("doy") * (_TAU / 365.25)).sin().alias("doy_sin"),
        (pl.col("doy") * (_TAU / 365.25)).cos().alias("doy_cos"),
    )


def add_multimodel_features(df: pl.DataFrame) -> pl.DataFrame:
    model_cols = [f"t_{m}" for m in config.MODELS if f"t_{m}" in df.columns]
    return df.with_columns(
        pl.mean_horizontal(model_cols).alias("t_mean"),
        pl.concat_list(model_cols).list.median().alias("t_median"),
        pl.concat_list(model_cols).list.std(ddof=0).alias("t_spread"),
        pl.min_horizontal(model_cols).alias("t_min"),
        pl.max_horizontal(model_cols).alias("t_max"),
    )


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Physically motivated interactions among the auxiliary predictors.

    Main effects of cloud/wind/humidity did not improve held-out skill on their
    own; the hypothesis is that they matter in *combination* (e.g. clear + calm
    night → strong radiative cooling). Only built when the inputs are present.
    """
    cols = set(df.columns)
    exprs = []
    # clear-sky night cooling: clear (low cloud) at night (low solar radiation)
    if {"cloud_mean", "rad_mean"} <= cols:
        night = (pl.col("rad_mean") < 1.0).cast(pl.Float64)
        exprs.append(((100.0 - pl.col("cloud_mean")) * night).alias("clearnight_cool"))
    # cloud x wind: cloud cover modulated by mixing
    if {"cloud_mean", "wind_mean"} <= cols:
        exprs.append((pl.col("cloud_mean") * pl.col("wind_mean")).alias("cloud_x_wind"))
    # cloud x time-of-day (cloud effect flips sign day vs night)
    if {"cloud_mean", "hour_cos"} <= cols:
        exprs.append((pl.col("cloud_mean") * pl.col("hour_cos")).alias("cloud_x_hourcos"))
    # humidity at night (fog / damp cooling)
    if {"rh_mean", "rad_mean"} <= cols:
        night = (pl.col("rad_mean") < 1.0).cast(pl.Float64)
        exprs.append((pl.col("rh_mean") * night).alias("rh_night"))
    return df.with_columns(exprs) if exprs else df


def add_recent_bias(
    df: pl.DataFrame, obs: pl.DataFrame, model: str, window: int = 30
) -> pl.DataFrame:
    col = f"t_{model}"
    out_name = f"recent_bias_{model}"
    work = (
        df.join(obs.rename({"t_obs": "_obs_truth"}), on="valid_time", how="left")
        .with_columns((pl.col(col) - pl.col("_obs_truth")).alias("_err"))
        .sort("issue_time")
    )
    # rolling mean over prior rows only (shift(1) excludes the current issue_time)
    work = work.with_columns(
        pl.col("_err")
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over("lead_time_h")
        .alias(out_name)
    )
    return work.drop(["_obs_truth", "_err"])
