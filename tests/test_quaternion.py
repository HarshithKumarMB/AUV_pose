"""Quaternion algebra invariants."""

import numpy as np
import pytest

from auv_pose.smoothing.quaternion import (
    quat_conjugate,
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
    skew,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def random_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    return [quat_normalize(rng.normal(size=4)) for _ in range(n)]


def assert_same_rotation(q, r):
    """Compare up to sign: q and -q are the same rotation."""
    if np.dot(q, r) < 0:
        r = -r
    np.testing.assert_allclose(q, r, atol=1e-9)


@pytest.mark.parametrize("q", random_quats(10))
def test_rotmat_roundtrip(q):
    assert_same_rotation(q, rotmat_to_quat(quat_to_rotmat(q)))


@pytest.mark.parametrize("q", random_quats(10))
def test_quat_to_rotmat_is_a_rotation(q):
    R = quat_to_rotmat(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotmat_roundtrip_at_180_degrees():
    """The trace <= 0 branch: a half-turn about each axis."""
    for axis in range(3):
        R = -np.eye(3)
        R[axis, axis] = 1.0
        assert_same_rotation(rotmat_to_quat(R), rotmat_to_quat(quat_to_rotmat(rotmat_to_quat(R))))
        np.testing.assert_allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-9)


def test_multiply_is_associative():
    a, b, c = random_quats(3, seed=1)
    left = quat_multiply(quat_multiply(a, b), c)
    right = quat_multiply(a, quat_multiply(b, c))
    np.testing.assert_allclose(left, right, atol=1e-9)


def test_multiply_is_not_commutative():
    a, b = random_quats(2, seed=2)
    assert not np.allclose(quat_multiply(a, b), quat_multiply(b, a))


@pytest.mark.parametrize("q", random_quats(5, seed=3))
def test_conjugate_is_the_inverse(q):
    np.testing.assert_allclose(quat_multiply(q, quat_conjugate(q)), IDENTITY, atol=1e-9)


@pytest.mark.parametrize("q", random_quats(5, seed=4))
def test_composition_matches_matrix_product(q):
    """R(q * r) == R(q) @ R(r): the algebra agrees with the matrices."""
    r = random_quats(1, seed=5)[0]
    np.testing.assert_allclose(
        quat_to_rotmat(quat_multiply(q, r)),
        quat_to_rotmat(q) @ quat_to_rotmat(r),
        atol=1e-9,
    )


def test_gyro_zero_rate_is_identity():
    np.testing.assert_array_equal(quat_from_gyro(np.zeros(3), 0.1), IDENTITY)


def test_gyro_below_threshold_is_identity():
    np.testing.assert_array_equal(quat_from_gyro([1e-12, 0.0, 0.0], 1e-3), IDENTITY)


def test_gyro_quarter_turn_about_z():
    q = quat_from_gyro([0.0, 0.0, np.pi / 2], 1.0)
    R = quat_to_rotmat(q)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_gyro_integrates_over_many_small_steps():
    """A full turn in 1000 steps equals a full turn in one."""
    omega = np.array([0.0, 0.0, 2 * np.pi])
    dt = 1.0 / 1000
    q = IDENTITY
    for _ in range(1000):
        q = quat_normalize(quat_multiply(q, quat_from_gyro(omega, dt)))
    np.testing.assert_allclose(quat_to_rotmat(q), np.eye(3), atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_skew_is_the_cross_product(seed):
    rng = np.random.default_rng(seed)
    w, v = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(skew(w) @ v, np.cross(w, v), atol=1e-12)


def test_skew_is_antisymmetric():
    w = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(skew(w).T, -skew(w))


def test_normalize_rejects_zero():
    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))
