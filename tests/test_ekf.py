"""Constant-velocity filter and RTS smoother.

Several of these pin the fixes made while extracting this class from ``way.py``:
the smoother could not run at all, and simultaneous measurements were applied as
two separate updates per predict.
"""

import numpy as np
import pytest

from auv_pose.smoothing.ekf import ConstantVelocityEKF

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])
DEPTH_H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


def test_predict_matches_hand_computed_step():
    """x = x0 + v0 dt + a dt^2 / 2, v = v0 + a dt."""
    ekf = ConstantVelocityEKF(x0=[1.0, 2.0, 3.0, 0.5, 0.0, -0.5])
    ekf.predict([0.0, 2.0, 0.0], dt=2.0)

    np.testing.assert_allclose(ekf.position, [2.0, 6.0, 2.0])
    np.testing.assert_allclose(ekf.velocity, [0.5, 4.0, -0.5])


def test_predict_grows_uncertainty():
    ekf = ConstantVelocityEKF()
    before = np.trace(ekf.P)
    ekf.predict(np.zeros(3), dt=0.1)
    assert np.trace(ekf.P) > before


def test_update_shrinks_uncertainty():
    ekf = ConstantVelocityEKF()
    ekf.predict(np.zeros(3), dt=0.1)
    before = np.trace(ekf.P)
    ekf.update([0.0, 0.0, 0.0], POSITION_H, np.eye(3) * 0.5)
    assert np.trace(ekf.P) < before


def test_update_keeps_covariance_symmetric_positive_definite():
    ekf = ConstantVelocityEKF()
    for _ in range(20):
        ekf.predict([0.1, 0.0, 0.0], dt=0.05)
        ekf.update([0.0, 0.0, 1.0], POSITION_H, np.eye(3) * 0.25)
    np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


def test_perfect_measurement_pulls_state_to_it():
    ekf = ConstantVelocityEKF(x0=[10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    ekf.predict(np.zeros(3), dt=0.1)
    ekf.update([0.0, 0.0, 0.0], POSITION_H, np.eye(3) * 1e-9)
    np.testing.assert_allclose(ekf.position, [0.0, 0.0, 0.0], atol=1e-4)


def test_stacked_update_equals_sequential_updates():
    """Two independent observations, stacked or applied in turn, agree.

    This is why navigate.py stacks its two depth observations into one update
    instead of calling update() twice per predict, as way.py did.
    """
    z1, z2 = 4.0, 6.0
    r1, r2 = 0.5, 2.0

    stacked = ConstantVelocityEKF(x0=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    stacked.predict(np.zeros(3), dt=0.1)
    stacked.update(
        [z1, z2],
        np.vstack([DEPTH_H, DEPTH_H]),
        np.diag([r1, r2]),
    )

    sequential = ConstantVelocityEKF(x0=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    sequential.predict(np.zeros(3), dt=0.1)
    sequential.update([z1], DEPTH_H, np.array([[r1]]))
    sequential.update([z2], DEPTH_H, np.array([[r2]]))

    np.testing.assert_allclose(stacked.x, sequential.x, atol=1e-10)
    np.testing.assert_allclose(stacked.P, sequential.P, atol=1e-10)


def test_depth_only_update_leaves_horizontal_position_untouched():
    """With H measuring z alone, x and y are unobservable."""
    ekf = ConstantVelocityEKF(x0=[3.0, -4.0, 0.0, 0.0, 0.0, 0.0])
    ekf.predict(np.zeros(3), dt=0.1)
    ekf.update([100.0], DEPTH_H, np.array([[0.1]]))
    np.testing.assert_allclose(ekf.position[:2], [3.0, -4.0], atol=1e-12)


def _run(n=25, seed=0):
    """A short linear-Gaussian run, returning the filter and the true positions."""
    rng = np.random.default_rng(seed)
    dt = 0.1
    ekf = ConstantVelocityEKF(
        x0=np.zeros(6), P0=np.eye(6), accel_process_sigma=0.5, keep_history=True
    )

    truth = []
    position = np.zeros(3)
    velocity = np.zeros(3)
    for _ in range(n):
        accel = rng.normal(scale=0.5, size=3)
        velocity = velocity + accel * dt
        position = position + velocity * dt
        truth.append(position.copy())

        ekf.predict(accel, dt)
        ekf.update(position + rng.normal(scale=0.3, size=3), POSITION_H, np.eye(3) * 0.09)

    return ekf, np.array(truth)


def test_rts_smoother_runs():
    """Regression: this raised IndexError, because predicted_* was never filled."""
    ekf, truth = _run()
    smoothed_x, smoothed_P = ekf.rts_smooth()
    assert len(smoothed_x) == len(truth) + 1  # +1 for the seeded initial state
    assert len(smoothed_P) == len(smoothed_x)
    assert all(np.all(np.isfinite(x)) for x in smoothed_x)
    assert all(np.all(np.isfinite(P)) for P in smoothed_P)


def test_rts_smoother_is_at_least_as_accurate_as_the_filter():
    ekf, truth = _run(n=40, seed=3)
    smoothed_x, _ = ekf.rts_smooth()

    filtered = np.array([x[:3, 0] for x, _ in ekf.filtered[1:]])
    smoothed = np.array([x[:3, 0] for x in smoothed_x[1:]])

    filter_rmse = np.sqrt(((filtered - truth) ** 2).sum(axis=1).mean())
    smoother_rmse = np.sqrt(((smoothed - truth) ** 2).sum(axis=1).mean())

    assert smoother_rmse <= filter_rmse


def test_rts_smoother_reduces_uncertainty():
    ekf, _ = _run()
    _, smoothed_P = ekf.rts_smooth()
    filtered_P = [P for _, P in ekf.filtered]

    # The final step is shared; every earlier step gains from future measurements.
    for k in range(len(smoothed_P) - 1):
        assert np.trace(smoothed_P[k]) <= np.trace(filtered_P[k]) + 1e-9


def test_rts_smoother_needs_history():
    ekf = ConstantVelocityEKF(keep_history=False)
    ekf.predict(np.zeros(3), dt=0.1)
    with pytest.raises(RuntimeError, match="keep_history"):
        ekf.rts_smooth()


def test_double_predict_is_rejected_immediately():
    """Two predicts with no update between would silently drop a step."""
    ekf = ConstantVelocityEKF(keep_history=True)
    ekf.predict(np.zeros(3), dt=0.1)
    with pytest.raises(RuntimeError, match="two predict"):
        ekf.predict(np.zeros(3), dt=0.1)


def test_smoothing_a_dangling_predict_is_rejected():
    ekf = ConstantVelocityEKF(keep_history=True)
    ekf.predict(np.zeros(3), dt=0.1)
    with pytest.raises(RuntimeError, match="not aligned"):
        ekf.rts_smooth()


def test_history_is_off_by_default():
    """A long run should not accumulate copies nothing reads."""
    ekf = ConstantVelocityEKF()
    ekf.predict(np.zeros(3), dt=0.1)
    ekf.update([0.0, 0.0, 0.0], POSITION_H, np.eye(3))
    with pytest.raises(RuntimeError, match="keep_history"):
        ekf.filtered


def test_update_rejects_mismatched_shapes():
    ekf = ConstantVelocityEKF()
    with pytest.raises(ValueError):
        ekf.update([1.0, 2.0], DEPTH_H, np.array([[1.0]]))
