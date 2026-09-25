"""Registering survey passes to each other before they are mapped together.

Each pass carries its own surface-fix error for its whole length: one
horizontal offset. Where passes overlap they disagree by it, and a map of all
of them smears every edge. This finds each pass's shift, and a vertical bias,
that makes it agree with the others -- crossover adjustment. Only relief pins a
shift, so flat seabed is refused rather than fitted to noise. The shifts are
relative and returned summing to zero.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Soundings that must fall on the reference surface for a shift to be fitted.
MIN_OVERLAP = 50


@dataclass(frozen=True)
class Grid:
  """A gridded surface: median elevation per cell, ``nan`` where empty.

  :param origin: World ``(x, y)`` of cell ``(0, 0)``'s centre.
  :param cell: Cell side, metres.
  :param z: Elevation, ``(ny, nx)``.
  """

  origin: NDArray[np.float64]
  cell: float
  z: NDArray[np.float64]

  def sample(
    self, points: NDArray[np.float64]
  ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bilinear elevation and its gradient at ``points``, ``nan`` off the grid."""
    u = (points - self.origin) / self.cell
    i0 = np.floor(u).astype(int)
    f = u - i0
    ny, nx = self.z.shape
    ok = (
      (i0[:, 0] >= 0)
      & (i0[:, 0] < nx - 1)
      & (i0[:, 1] >= 0)
      & (i0[:, 1] < ny - 1)
    )
    ix = np.clip(i0[:, 0], 0, nx - 2)
    iy = np.clip(i0[:, 1], 0, ny - 2)
    z00, z10 = self.z[iy, ix], self.z[iy, ix + 1]
    z01, z11 = self.z[iy + 1, ix], self.z[iy + 1, ix + 1]
    fx, fy = f[:, 0], f[:, 1]
    value = (
      z00 * (1 - fx) * (1 - fy)
      + z10 * fx * (1 - fy)
      + z01 * (1 - fx) * fy
      + z11 * fx * fy
    )
    dx = ((z10 - z00) * (1 - fy) + (z11 - z01) * fy) / self.cell
    dy = ((z01 - z00) * (1 - fx) + (z11 - z10) * fx) / self.cell
    value = np.where(ok, value, np.nan)
    return value, np.where(ok[:, None], np.stack([dx, dy], axis=1), np.nan)


def grid_surface(points: ArrayLike, z: ArrayLike, cell: float = 1.0) -> Grid:
  """Median elevation per cell of a set of soundings."""
  points = np.asarray(points, dtype=float)
  z = np.asarray(z, dtype=float)
  low = np.floor(points.min(axis=0) / cell) * cell
  index = np.floor((points - low) / cell).astype(int)
  shape = index.max(axis=0) + 1
  flat = index[:, 1] * shape[0] + index[:, 0]
  order = np.argsort(flat, kind="stable")
  cells, starts = np.unique(flat[order], return_index=True)
  medians = np.array(
    [np.median(part) for part in np.split(z[order], starts[1:])]
  )
  grid = np.full(int(shape[0] * shape[1]), np.nan)
  grid[cells] = medians
  return Grid(
    low + 0.5 * cell, cell, grid.reshape(int(shape[1]), int(shape[0]))
  )


def shift_to(
  points: NDArray[np.float64],
  z: NDArray[np.float64],
  reference: Grid,
  search: float = 6.0,
  step: float = 0.5,
  iterations: int = 15,
) -> tuple[NDArray[np.float64], float]:
  """Horizontal shift and vertical bias making ``points`` agree with a surface.

  A coarse search over ``+-search`` metres finds the basin -- offsets of a few
  metres are far outside where a gradient step alone would converge -- then
  Gauss-Newton refines it, with Huber weights so a few mismatched edges do not
  drag the answer.

  :return: ``(shift, bias)``: add ``shift`` to the points' ``(x, y)`` and
      ``bias`` to their ``z`` to match the reference.
  :raises ValueError: If fewer than ``MIN_OVERLAP`` points fall on the
      reference at every shift searched.
  """
  best, best_cost = np.zeros(2), np.inf
  candidates = np.arange(-search, search + step / 2, step)
  for sx in candidates:
    for sy in candidates:
      value, _ = reference.sample(points + [sx, sy])
      residual = z - value
      good = np.isfinite(residual)
      if good.sum() < MIN_OVERLAP:
        continue
      centred = residual[good] - np.median(residual[good])
      cost = float(np.median(np.abs(centred)))
      if cost < best_cost:
        best, best_cost = np.array([sx, sy]), cost

  if not np.isfinite(best_cost):
    raise ValueError(
      f"fewer than {MIN_OVERLAP} soundings overlap the other passes"
    )

  shift, bias = best.astype(float), 0.0
  for _ in range(iterations):
    value, slope = reference.sample(points + shift)
    residual = z + bias - value
    good = np.isfinite(residual) & np.isfinite(slope).all(axis=1)
    if good.sum() < MIN_OVERLAP:
      break
    r, g = residual[good], slope[good]
    scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    weight = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(r), 1e-12))
    # Residual r = z + b - S(x + s): d r / d s = -grad S, d r / d b = 1.
    jacobian = np.column_stack([-g, np.ones(len(r))])
    normal = jacobian.T @ (weight[:, None] * jacobian)
    update = np.linalg.solve(normal, -jacobian.T @ (weight * r))
    shift, bias = shift + update[:2], bias + float(update[2])
    if np.max(np.abs(update[:2])) < 1e-3:
      break
  return shift, bias


def register_passes(
  passes: Sequence[tuple[ArrayLike, ArrayLike]],
  cell: float = 1.0,
  rounds: int = 3,
  search: float = 6.0,
  samples: int = 20_000,
  seed: int = 0,
  tolerance: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Shift each pass to agree with the others, the shifts summing to zero.

  Each round registers every pass's own soundings against the gridded surface
  of all the others as currently aligned, so an early bad pass does not anchor
  the rest. Soundings, not cell centres, are registered: a grid's cell centres
  would quantise the shift to the cell.

  :param passes: ``(points, z)`` per pass, ``(n, 2)`` and ``(n,)``.
  :param cell: Grid cell for the reference surfaces, metres.
  :param rounds: Sweeps over all passes.
  :param search: Coarse search half-width, metres; only in the first round.
  :param samples: Soundings per pass used to fit its shift.
  :param seed: Seed for choosing them.
  :param tolerance: Largest disagreement, metres, allowed between shifts fitted
      on two random halves of the data.
  :return: ``(shifts, biases)``: ``(P, 2)`` horizontal and ``(P,)`` vertical,
      each summing to zero -- add them to a pass's soundings.
  :raises ValueError: If a pass's shift is not pinned down: over flat seabed,
      or relief that runs along one axis, the fit locks onto noise, and two
      halves of the data disagree by metres where relief makes them agree to
      centimetres.
  """
  if len(passes) < 2:
    raise ValueError("registration needs at least two passes")
  rng = np.random.default_rng(seed)
  passes = [(np.asarray(xy, float), np.asarray(z, float)) for xy, z in passes]
  chosen = [
    rng.choice(len(z), min(samples, len(z)), replace=False) for _, z in passes
  ]
  shifts = np.zeros((len(passes), 2))
  biases = np.zeros(len(passes))

  for sweep in range(rounds):
    for p, (xy, z) in enumerate(passes):
      others = [q for q in range(len(passes)) if q != p]
      reference = grid_surface(
        np.concatenate([passes[q][0] + shifts[q] for q in others]),
        np.concatenate([passes[q][1] + biases[q] for q in others]),
        cell,
      )
      shifts[p], biases[p] = shift_to(
        xy[chosen[p]],
        z[chosen[p]],
        reference,
        search=search if sweep == 0 else 1.0,
        step=0.5 if sweep == 0 else 0.25,
      )
    shifts -= shifts.mean(axis=0)
    biases -= biases.mean()

  _check_observable(
    passes, shifts, biases, cell, search, samples, rng, tolerance
  )
  return shifts, biases


def _check_observable(
  passes, shifts, biases, cell, search, samples, rng, tolerance
):
  """Refit each pass's shift on two halves of every pass; they must agree."""
  halves = [rng.permutation(len(z)) for _, z in passes]
  for p in range(len(passes)):
    others = [q for q in range(len(passes)) if q != p]
    fits = []
    for half in (0, 1):
      part = [order[half::2] for order in halves]
      reference = grid_surface(
        np.concatenate([passes[q][0][part[q]] + shifts[q] for q in others]),
        np.concatenate([passes[q][1][part[q]] + biases[q] for q in others]),
        cell,
      )
      mine = part[p][:samples]
      fits.append(
        shift_to(passes[p][0][mine], passes[p][1][mine], reference, search)[0]
      )
    disagreement = np.abs(fits[0] - fits[1])
    if np.any(disagreement > tolerance):
      axes = " and ".join(
        "xy"[i] for i in np.flatnonzero(disagreement > tolerance)
      )
      raise ValueError(
        f"pass {p} cannot be registered in {axes}: two halves of the data "
        f"disagree by {np.round(disagreement, 2)} m; the seabed lacks relief "
        "across that axis"
      )
