import numpy as np
from wetter.models.conformal import SplitConformal, ConformalizedQuantile


def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    y_cal = rng.normal(0, 1, 2000)
    point_cal = np.zeros(2000)
    sc = SplitConformal().calibrate(y_cal, point_cal)

    y_test = rng.normal(0, 1, 5000)
    point_test = np.zeros(5000)
    lo, hi = sc.interval(point_test, alpha=0.2)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    assert abs(covered - 0.8) < 0.03  # ~80% nominal coverage
    assert np.all(hi >= lo)


def test_cqr_fixes_undercoverage():
    rng = np.random.default_rng(0)
    # quantile model is far too narrow: predicts [-0.5, 0.5] for N(0,1) data
    # (raw coverage ~0.38). CQR must widen it to ~0.80.
    y_cal = rng.normal(0, 1, 3000)
    qlo_cal = np.full(3000, -0.5)
    qhi_cal = np.full(3000, 0.5)
    cqr = ConformalizedQuantile().calibrate(y_cal, qlo_cal, qhi_cal)

    y_test = rng.normal(0, 1, 5000)
    lo, hi = cqr.interval(np.full(5000, -0.5), np.full(5000, 0.5), alpha=0.2)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    assert abs(covered - 0.8) < 0.03
    assert np.all(hi >= lo)
