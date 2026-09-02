"""Attitude determination.

The two Wahba solvers are independent implementations, so their agreement is
meaningful evidence that both are right -- hence the assertion rather than a
printed diagnostic.
"""

import numpy as np
import pytest

from auv_pose.smoothing.attitude import (
    AttitudeFilter,
    accel_weight,
    wahba_davenport,
    wahba_svd,
)
from auv_pose.smoothing.quaternion import quat_to_rotmat, rotmat_to_quat

DOWN = np.array([0.0, 0.0, 1.0])


def random_rotation(seed):
    """A uniformly random rotation matrix, via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def same_rotation(q, r):
    return np.allclose(q, r, atol=1e-8) or np.allclose(q, -r, atol=1e-8)


@pytest.mark.parametrize("seed", range(8))
def test_solvers_agree(seed):
    """SVD and Davenport must produce the same attitude."""
    R = random_rotation(seed)
    v_ref = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    v_body = [R.T @ v for v in v_ref]
    weights = [0.5, 0.5]

    assert same_rotation(
        wahba_svd(v_body, v_ref, weights),
        wahba_davenport(v_body, v_ref, weights),
    )


@pytest.mark.parametrize("solver", [wahba_svd, wahba_davenport])
@pytest.mark.parametrize("seed", range(5))
def test_recovers_a_known_rotation(solver, seed):
    """Two non-parallel observations determine attitude uniquely."""
    R = random_rotation(seed)
    v_ref = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    v_body = [R.T @ v for v in v_ref]

    recovered = quat_to_rotmat(solver(v_body, v_ref, [1.0, 1.0]))
    np.testing.assert_allclose(recovered, R, atol=1e-8)


@pytest.mark.parametrize("solver", [wahba_svd, wahba_davenport])
def test_identity_observations_give_identity(solver):
    v = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    assert same_rotation(solver(v, v, [1.0, 1.0]), rotmat_to_quat(np.eye(3)))


def test_weights_bias_toward_the_trusted_observation():
    """Given contradictory observations, the heavier one wins."""
    v_ref = [DOWN, DOWN]
    tilted = quat_to_rotmat(rotmat_to_quat(random_rotation(2))) @ DOWN
    v_body = [DOWN, tilted / np.linalg.norm(tilted)]

    trust_first = quat_to_rotmat(wahba_svd(v_body, v_ref, [0.99, 0.01])) @ DOWN
    trust_second = quat_to_rotmat(wahba_svd(v_body, v_ref, [0.01, 0.99])) @ DOWN

    assert np.dot(trust_first, v_body[0]) > np.dot(trust_second, v_body[0])


def test_accel_weight_peaks_at_one_g():
    assert accel_weight([0.0, 0.0, 1.0]) == pytest.approx(1.0)
    assert accel_weight([0.0, 0.0, 2.0]) < accel_weight([0.0, 0.0, 1.2])
    assert accel_weight([0.0, 0.0, 1.2]) < 1.0


def test_filter_stays_level_when_at_rest():
    """Gravity-only observations with no rotation: attitude must not wander."""
    filt = AttitudeFilter(kp=1.0, ki=0.05)
    for _ in range(200):
        filt.update(np.zeros(3), [DOWN, DOWN], dt=0.01)
    np.testing.assert_allclose(filt.rotation, np.eye(3), atol=1e-6)


def test_filter_tracks_pure_gyro_rotation():
    """With no usable accelerometer, the estimate is the gyro integral."""
    filt = AttitudeFilter()
    omega = np.array([0.0, 0.0, np.pi / 2])
    for _ in range(100):
        filt.update(omega, [np.zeros(3)], dt=0.01)

    np.testing.assert_allclose(
        filt.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6
    )


def test_filter_corrects_an_initial_tilt_error():
    """Starting wrong, level accelerometers should pull the estimate back."""
    tilt = rotmat_to_quat(quat_to_rotmat([np.cos(0.1), np.sin(0.1), 0.0, 0.0]))
    filt = AttitudeFilter(kp=1.0, q0=tilt)

    start = abs(filt.rotation[2, 2])
    for _ in range(500):
        filt.update(np.zeros(3), [DOWN], dt=0.01)

    assert abs(filt.rotation[2, 2]) > start
    assert abs(filt.rotation[2, 2]) == pytest.approx(1.0, abs=1e-4)


def test_cross_check_reports_solver_agreement():
    """The diagnostic is opt-in; both solvers must still agree when it is on."""
    filt = AttitudeFilter(cross_check=True)
    filt.update(np.zeros(3), [DOWN, DOWN], dt=0.01)
    assert filt.solver_disagreement < 1e-6
