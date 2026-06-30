import polars as pl
from wetter.models.bias_correction import BiasCorrector


def test_bias_correction_removes_systematic_offset():
    # model always +2 warm in (lead 24, hour 0, month 1)
    train = pl.DataFrame(
        {
            "t_icon_d2": [12.0, 13.0, 14.0],
            "t_obs": [10.0, 11.0, 12.0],
            "lead_time_h": [24, 24, 24],
            "hour": [0, 0, 0],
            "month": [1, 1, 1],
        }
    )
    bc = BiasCorrector().fit(train, "icon_d2")
    test = pl.DataFrame({"t_icon_d2": [20.0], "lead_time_h": [24], "hour": [0], "month": [1]})
    assert abs(bc.predict(test)[0] - 18.0) < 1e-9  # 20 - learned(+2)
