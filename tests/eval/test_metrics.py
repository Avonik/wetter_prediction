import numpy as np
from wetter.eval import metrics as M


def test_mae_rmse_bias():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, -1.0, 2.0])
    assert M.mae(y, p) == (1 + 1 + 2) / 3
    assert abs(M.rmse(y, p) - np.sqrt((1 + 1 + 4) / 3)) < 1e-12
    assert abs(M.bias(y, p) - (1 - 1 + 2) / 3) < 1e-12


def test_skill_score_signs():
    assert M.skill_score(0.5, 1.0) == 0.5  # half the error of ref -> +0.5
    assert M.skill_score(2.0, 1.0) == -1.0  # worse than ref -> negative


def test_crps_gaussian_perfect_sharp_is_small():
    y = np.array([5.0])
    mu = np.array([5.0])
    sigma = np.array([1e-3])
    assert M.crps_gaussian(y, mu, sigma)[0] < 1e-3


def test_crps_gaussian_known_value():
    # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi)
    val = M.crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert abs(val - 0.23369497) < 1e-6
