"""A linear Kalman filter and RTS smoother: the known-exact answer.

On a linear-Gaussian problem the unscented filter and smoother must reproduce
these to round-off, which is what the tests that import this check.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class GaussianState:
  mean: np.ndarray
  cov: np.ndarray


@dataclass(frozen=True, eq=False)
class Measurement:
  z: np.ndarray
  H: np.ndarray
  R: np.ndarray


@dataclass(frozen=True, eq=False)
class Step:
  prior: GaussianState
  posterior: GaussianState
  transition: np.ndarray


class ConstantVelocityKF:
  """State ``[x, y, z, vx, vy, vz]``, driven by world-frame acceleration."""

  def __init__(self, accel_process_sigma: float = 0.5) -> None:
    self.accel_process_sigma = accel_process_sigma
    self.history: list[Step] = []

  @staticmethod
  def initial(position, velocity=None, cov=None) -> GaussianState:
    velocity = np.zeros(3) if velocity is None else velocity
    mean = np.concatenate(
      [np.asarray(position, float), np.asarray(velocity, float)]
    )
    return GaussianState(
      mean, np.eye(6) if cov is None else np.asarray(cov, float)
    )

  def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transition ``F``, control ``B`` and process noise ``Q`` for ``dt``."""
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    B = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
    return F, B, self.accel_process_sigma**2 * B @ B.T

  def step(self, state, accel, dt, observations=()) -> GaussianState:
    F, B, Q = self.matrices(dt)
    prior = GaussianState(
      F @ state.mean + B @ np.asarray(accel, float), F @ state.cov @ F.T + Q
    )
    posterior = prior
    for obs in observations:
      H, R = np.atleast_2d(obs.H), np.atleast_2d(obs.R)
      gain = posterior.cov @ H.T @ np.linalg.inv(H @ posterior.cov @ H.T + R)
      joseph = np.eye(6) - gain @ H
      posterior = GaussianState(
        posterior.mean + gain @ (np.asarray(obs.z, float) - H @ posterior.mean),
        joseph @ posterior.cov @ joseph.T + gain @ R @ gain.T,
      )
    self.history.append(Step(prior, posterior, F))
    return posterior


def rts_smooth(
  initial: GaussianState, history: list[Step]
) -> list[GaussianState]:
  """Smoothed beliefs, oldest first, including the initial one."""
  filtered = [initial] + [step.posterior for step in history]
  smoothed = [filtered[-1]] * len(filtered)
  for k in range(len(history) - 1, -1, -1):
    step = history[k]
    gain = filtered[k].cov @ step.transition.T @ np.linalg.inv(step.prior.cov)
    smoothed[k] = GaussianState(
      filtered[k].mean + gain @ (smoothed[k + 1].mean - step.prior.mean),
      filtered[k].cov + gain @ (smoothed[k + 1].cov - step.prior.cov) @ gain.T,
    )
  return smoothed
