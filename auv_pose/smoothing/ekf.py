"""Constant-velocity Kalman filter with an RTS smoother.

State is ``[x, y, z, vx, vy, vz]`` in the world frame, driven by world-frame
acceleration as a control input. The dynamics are linear, so this is an ordinary
Kalman filter; the name reflects its role in a pipeline where the nonlinearity
lives in the measurement construction rather than in the filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ConstantVelocityEKF"]

_STATE_DIM = 6


@dataclass(frozen=True)
class _Step:
    """One predict/update cycle, as the RTS smoother needs it.

    Holding these together makes the alignment the smoother depends on structural,
    rather than an invariant across parallel lists.
    """

    prior_x: NDArray
    prior_P: NDArray
    transition: NDArray
    posterior_x: NDArray
    posterior_P: NDArray


class ConstantVelocityEKF:
    """Position/velocity filter with acceleration as a control input.

    Args:
        x0: Initial state, 6 elements. Zero if omitted.
        P0: Initial covariance. Identity if omitted.
        accel_process_sigma: Standard deviation of unmodelled acceleration, m/s^2.
            Shapes the process noise as ``sigma^2 * B B^T``, so it enters position
            and velocity consistently for the timestep.
        keep_history: Retain the per-step quantities :meth:`rts_smooth` needs. Off
            by default: a long run accumulates two 6x6 copies per step, which is
            wasted unless something actually smooths.

    Note:
        With ``keep_history``, each step must be exactly one :meth:`predict` then
        one :meth:`update`; anything else raises. To fold several simultaneous
        observations into a step, stack them into a single ``update`` call rather
        than calling it twice -- see :meth:`update`.
    """

    def __init__(
        self,
        x0: ArrayLike | None = None,
        P0: ArrayLike | None = None,
        accel_process_sigma: float = 0.5,
        keep_history: bool = False,
    ) -> None:
        self.x = (
            np.zeros((_STATE_DIM, 1))
            if x0 is None
            else np.asarray(x0, dtype=float).reshape(_STATE_DIM, 1)
        )
        self.P = np.eye(_STATE_DIM) if P0 is None else np.asarray(P0, dtype=float)
        self.accel_process_sigma = accel_process_sigma
        self.keep_history = keep_history

        self._initial: tuple[NDArray, NDArray] = (self.x.copy(), self.P.copy())
        self._steps: list[_Step] = []
        self._pending: tuple[NDArray, NDArray, NDArray] | None = None

        # dt is constant across a run, so F, B and Q are computed once and reused.
        self._cached_dt: float | None = None
        self._cached: tuple[NDArray, NDArray, NDArray] | None = None

    def _matrices(self, dt: float) -> tuple[NDArray, NDArray, NDArray]:
        """``(F, B, Q)`` for a constant-velocity step of length ``dt``."""
        if dt != self._cached_dt:
            F = np.eye(_STATE_DIM)
            F[:3, 3:] = np.eye(3) * dt

            B = np.zeros((_STATE_DIM, 3))
            B[:3, :] = np.eye(3) * (0.5 * dt**2)
            B[3:, :] = np.eye(3) * dt

            self._cached_dt = dt
            self._cached = (F, B, self.accel_process_sigma**2 * (B @ B.T))

        assert self._cached is not None
        return self._cached

    def predict(self, accel_world: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Propagate the state by ``dt`` under world-frame acceleration.

        Args:
            accel_world: Linear acceleration in the world frame, m/s^2, with gravity
                already removed. See :mod:`auv_pose.smoothing` for the convention.
            dt: Interval, seconds.
        """
        F, B, Q = self._matrices(dt)
        u = np.asarray(accel_world, dtype=float).reshape(3, 1)

        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + Q

        if self.keep_history:
            if self._pending is not None:
                raise RuntimeError(
                    "two predict() calls without an update() between them; the "
                    "smoother needs exactly one of each per step"
                )
            self._pending = (self.x.copy(), self.P.copy(), F)

        return self.x

    def update(
        self,
        z: ArrayLike,
        H: ArrayLike,
        R: ArrayLike,
    ) -> NDArray[np.float64]:
        """Correct the state with a measurement.

        Args:
            z: Measurement, ``m`` elements.
            H: Measurement model, ``m x 6``, mapping state to measurement space.
            R: Measurement noise covariance, ``m x m``.

        Stack simultaneous observations into one call rather than calling this twice
        per step. For independent measurements the two are algebraically equivalent,
        but the stacked form keeps the history aligned for :meth:`rts_smooth`, and it
        makes the full measurement model visible in one place.
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        R = np.atleast_2d(np.asarray(R, dtype=float))
        z = np.asarray(z, dtype=float).reshape(H.shape[0], 1)

        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        # Joseph form: stays symmetric positive-definite even when K is not the
        # exact optimal gain, which matters once R is tuned by hand.
        I_KH = np.eye(_STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        if self.keep_history:
            if self._pending is None:
                raise RuntimeError(
                    "update() without a preceding predict(); the smoother needs "
                    "exactly one of each per step"
                )
            prior_x, prior_P, F = self._pending
            self._pending = None
            self._steps.append(
                _Step(prior_x, prior_P, F, self.x.copy(), self.P.copy())
            )

        return self.x

    @property
    def position(self) -> NDArray[np.float64]:
        """Current position estimate."""
        return self.x[:3, 0].copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Current velocity estimate."""
        return self.x[3:, 0].copy()

    @property
    def filtered(self) -> list[tuple[NDArray, NDArray]]:
        """Posterior ``(x, P)`` per step, oldest first, including the initial state."""
        if not self.keep_history:
            raise RuntimeError("filtered history needs keep_history=True")
        return [self._initial] + [(s.posterior_x, s.posterior_P) for s in self._steps]

    def rts_smooth(self) -> tuple[list[NDArray], list[NDArray]]:
        """Rauch-Tung-Striebel fixed-interval smoother over the recorded run.

        Returns:
            ``(smoothed_x, smoothed_P)``, one entry per step, oldest first.

        Raises:
            RuntimeError: If history was disabled, or if a predict is left without
                its matching update -- the smoother needs one of each per step.
        """
        if not self.keep_history:
            raise RuntimeError("rts_smooth() needs keep_history=True")
        if self._pending is not None:
            raise RuntimeError(
                "history is not aligned: a predict() has no matching update(). "
                "Each step must be exactly one predict() followed by one update()."
            )

        posteriors = self.filtered
        n = len(posteriors)

        smoothed_x: list[NDArray] = [np.empty(0)] * n
        smoothed_P: list[NDArray] = [np.empty(0)] * n
        smoothed_x[-1], smoothed_P[-1] = posteriors[-1]

        for k in range(n - 2, -1, -1):
            x_filt, P_filt = posteriors[k]
            step = self._steps[k]  # the step leading from k to k + 1

            C = P_filt @ step.transition.T @ np.linalg.inv(step.prior_P)
            smoothed_x[k] = x_filt + C @ (smoothed_x[k + 1] - step.prior_x)
            smoothed_P[k] = P_filt + C @ (smoothed_P[k + 1] - step.prior_P) @ C.T

        return smoothed_x, smoothed_P
