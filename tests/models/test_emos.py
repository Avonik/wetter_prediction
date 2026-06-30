import numpy as np
import polars as pl
from wetter.models.emos import EMOS


def test_emos_recovers_offset_and_calibrated_spread():
    rng = np.random.default_rng(0)
    n = 2000
    t_mean = rng.normal(10, 5, n)
    spread = np.abs(rng.normal(1.0, 0.2, n))
    # truth = 1 + 1.0*t_mean + gaussian noise scaled by spread
    y = 1.0 + t_mean + rng.normal(0, 1, n) * spread
    df = pl.DataFrame({"t_mean": t_mean, "t_spread": spread, "t_obs": y})
    em = EMOS().fit(df)
    mu, sigma = em.predict(df)
    assert abs(np.mean(mu - y)) < 0.2
    assert np.mean(sigma) > 0
    z = (y - mu) / sigma
    assert 0.7 < np.std(z) < 1.3
