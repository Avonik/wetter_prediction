import polars as pl
from wetter.data import climatology as clim

PAYLOAD = {
    "hourly": {
        "time": ["2020-06-01T12:00", "2021-06-15T12:00", "2020-01-01T00:00"],
        "temperature_2m": [20.0, 22.0, -1.0],
    }
}


def test_compute_climatology_groups_month_hour():
    era5 = clim.parse_era5(PAYLOAD)
    c = clim.compute_climatology(era5)
    june_noon = c.filter((pl.col("month") == 6) & (pl.col("hour") == 12))["t_clim"][0]
    assert june_noon == 21.0  # mean(20, 22)
    assert c["t_clim"].dtype == pl.Float64
