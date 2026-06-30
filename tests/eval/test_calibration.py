import numpy as np
from wetter.eval import calibration as C


def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(0)
    n = 50000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y = rng.normal(0, 1, n)
    pit = C.pit_gaussian(y, mu, sigma)
    assert abs(pit.mean() - 0.5) < 0.01
    assert abs(pit.std() - (1 / np.sqrt(12))) < 0.01


def test_coverage_and_sharpness():
    y = np.array([0.0, 0.0, 5.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert abs(C.coverage(y, lo, hi) - 2 / 3) < 1e-9
    assert C.sharpness(lo, hi) == 2.0
