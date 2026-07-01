import numpy as np
from wetter.eval import rain_metrics as R


def test_brier_known_values():
    assert R.brier(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert R.brier(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.25


def test_brier_skill_score_signs():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(float)
    assert R.brier_skill_score(y, y) > 0.99  # perfect -> ~1
    base = np.full_like(y, y.mean())
    assert abs(R.brier_skill_score(y, base)) < 1e-9  # base rate -> 0


def test_reliability_curve_calibrated():
    rng = np.random.default_rng(1)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(float)  # perfectly calibrated by construction
    mean_pred, obs_freq, counts = R.reliability_curve(y, p, n_bins=10)
    ok = ~np.isnan(mean_pred)
    assert np.allclose(mean_pred[ok], obs_freq[ok], atol=0.03)
    assert counts.sum() == 50000
