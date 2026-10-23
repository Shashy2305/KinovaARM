"""calibration/rescale_wizard_result.py undoes a board-size mistake exactly (forward model of multi_camera_calibrate.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import rescale_wizard_result as rw  # noqa: E402


def wizard_estimate(x_true, p_w, k):
    """what the wizard writes when every board translation is scaled by k: x_est = p_w + k (b - p_w) - k (b - x_true) = (1-k) p_w + k x_true"""
    b = np.array([0.45, 0.10, -0.01])
    return p_w + k * (b - p_w) - k * (b - x_true)


def test_rescale_recovers_the_true_camera_position():
    x_true = np.array([1.02, -0.03, 0.77])
    p_w = np.array([0.40, 0.05, 0.50])
    k = 35.0 / 38.15
    x_est = wizard_estimate(x_true, p_w, k)
    assert np.linalg.norm(x_est - x_true) > 0.05                              # the mistake is several cm
    assert np.allclose(rw.rescale(x_est, p_w, 0.035, 0.03815), x_true, atol=1e-9)


def test_an_uncertain_wrist_position_matters_only_by_the_small_factor():
    x_true = np.array([0.41, 0.90, 0.95])
    p_w = np.array([0.40, 0.05, 0.50])
    k = 35.0 / 38.15
    x_est = wizard_estimate(x_true, p_w, k)
    wrong = rw.rescale(x_est, p_w + np.array([0.15, -0.15, 0.10]), 0.035, 0.03815)
    assert np.linalg.norm(wrong - x_true) < 0.03                              # 15 cm wrong wrist position -> ~1.4-2 cm
    assert np.linalg.norm(x_est - x_true) > 0.05                              # still much better than not correcting


def test_a_correct_size_changes_nothing():
    t = np.array([0.5, 0.4, 0.9])
    assert np.allclose(rw.rescale(t, [0.4, 0, 0.5], 0.0381, 0.0381), t)
