import polars as pl
from wetter.models import baselines as B


def test_baselines_read_columns():
    df = pl.DataFrame({"t_obs_at_issue": [1.0], "t_clim": [2.0], "t_icon_d2": [3.0]})
    assert B.predict_persistence(df)[0] == 1.0
    assert B.predict_climatology(df)[0] == 2.0
    assert B.predict_raw(df, "icon_d2")[0] == 3.0
