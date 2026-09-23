"""The Vecchia approximation: a Gaussian process that scales to a whole survey.

An exact GP over ``N`` soundings costs ``O(N^3)``, which a multibeam survey puts
out of reach within an afternoon of flying. Vecchia's approximation replaces the
joint density by a chain in which each sounding conditions on only its ``m``
nearest **predecessors** under some ordering::

    p(z) = prod_i p(z_i | z_{g(i)}),    g(i) subset of {1..i-1},  |g(i)| <= m

which is exact when ``g(i)`` is every predecessor and an approximation
otherwise. It is linear in ``N`` in both time and memory, and -- unlike an
inducing-point method -- it conditions *locally*, so fine structure survives
rather than being averaged into a few hundred global basis functions.

**Everything here works on the residual of a linear mean.** The map is
``z = phi(q)^T beta + zeta(q) + eps`` with ``phi(q) = (1, n, e)``; ``beta``
absorbs the regional depth and slope, and this module models ``zeta`` alone.
That is why there is no target standardisation anywhere: the mean does the job
the centring used to, and the per-axis lengthscales do the job the input scaler
used to, with the advantage that both come out in metres.

**float64, throughout, deliberately.** After decimation to 0.25 m against a
lengthscale of metres, a block of neighbouring soundings is strongly correlated
and its kernel matrix is correspondingly ill-conditioned -- measured condition
numbers around 3.7e6 at 0.25 m spacing. float32 carries about 1e7 of dynamic
range, so it sits at the edge of failing; float64 has room to spare and the
batched Cholesky is fast enough either way.

See Vecchia (1988) for the original, Katzfuss and Guinness (2021) for the
framework, and Rambelli and Sigrist (2026) for why this rather than the
alternatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from auv_pose.mapping.kernels import matern52, matern52_gradient
from auv_pose.mapping.ordering import maximin_order, ordered_neighbours

__all__ = [
  "DEFAULT_JITTER",
  "MeanBasis",
  "VecchiaHyperparameters",
  "VecchiaMap",
  "VecchiaStructure",
  "build_structure",
  "design_matrix",
  "draw",
  "fit_vecchia",
  "fit_vecchia_nigp",
  "initial_hyperparameters",
  "sparse_factor",
  "vecchia_loglik",
  "vecchia_reml",
  "whiten",
  "whiten_gram",
]


def _resolve_device(device: str | torch.device | None) -> torch.device:
  """CUDA when it is genuinely there, else the CPU.

  Mirrors :func:`auv_pose.mapping.svgp.resolve_device`; duplicated rather than
  imported so this module does not pull in gpytorch.
  """
  if device is not None:
    return torch.device(device)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_LOG_2PI = math.log(2.0 * math.pi)

#: Added to every kernel block's diagonal before factorising. A regulariser,
#: **not** a nugget with a physical meaning: soundings 0.25 m apart under a
#: metre-scale kernel have covariances indistinguishable from their variance,
#: and the Cholesky fails on the resulting singularity. Scaled by the amplitude
#: so it stays proportionate when the fit moves.
DEFAULT_JITTER = 1e-8


@dataclass(frozen=True)
class VecchiaStructure:
  """The part of the approximation that depends only on *where* the soundings are.

  Held apart from the hyperparameters on purpose. The ordering and the
  conditioning sets are functions of position alone, so they are computed once
  and reused across every step of the hyperparameter fit -- which is what makes
  each step cost one batched Cholesky rather than a neighbour search.

  :param points: Sounding positions **in the approximation's own order**,
      shape ``(N, 2)``, metres.
  :param neighbours: Indices into :attr:`points` of each sounding's ``m``
      nearest predecessors, shape ``(N, m)``, padded with ``-1`` where there
      are fewer. Rows below :attr:`n0` are unused.
  :param order: The permutation taking the caller's ordering to this one.
  :param n0: Soundings handled by a single dense Cholesky rather than by the
      chain. Must be at least ``m`` so the batched rows all have exactly ``m``
      neighbours and stay rectangular.
  """

  points: np.ndarray
  neighbours: np.ndarray
  order: np.ndarray
  n0: int

  @property
  def size(self) -> int:
    """Number of soundings."""
    return len(self.points)

  @property
  def conditioning(self) -> int:
    """``m``, the conditioning-set size."""
    return self.neighbours.shape[1]


def build_structure(
  points: ArrayLike,
  m: int = 30,
  n0: int | None = None,
  first: int | None = None,
  near: int | None = None,
) -> VecchiaStructure:
  """Order the soundings and find each one's conditioning set.

  :param points: Sounding positions, shape ``(N, 2)``, in any order.
  :param m: Conditioning-set size.
  :param n0: Size of the dense head block. Defaults to ``max(m, 64)``, capped
      at ``N``.
  :param first: Passed to :func:`~auv_pose.mapping.ordering.maximin_order`.
  :param near: How many of the ``m`` are nearest neighbours; the rest are
      spread across the ordering. See
      :func:`~auv_pose.mapping.ordering.ordered_neighbours`. Defaults to all
      nearest, which is right for prediction and wrong for fitting.
  :return: The structure, with ``points`` permuted into maximin order.

  Note:
      ``n0`` is not merely bookkeeping to keep the batched rows rectangular.
      The head block conditions each of its soundings on *all* its
      predecessors, so it is locally exact, and enlarging it buys accuracy for
      a dense factorisation of a few dozen points -- which costs nothing. The
      same observation is made in GraphGP (Dodge, Frank and Clark, 2026), who
      use a dense block of 100 "to avoid small initial batches and increase
      accuracy".
  """
  points = np.asarray(points, dtype=float)
  if points.ndim != 2 or points.shape[1] != 2:
    raise ValueError(f"expected (N, 2) points, got {points.shape}")
  if m < 1:
    raise ValueError(f"expected m >= 1, got {m}")

  size = len(points)
  if n0 is None:
    n0 = max(m, 64)
  n0 = min(max(n0, m), size)

  order = maximin_order(points, first=first)
  ordered = points[order]

  return VecchiaStructure(
    points=ordered,
    neighbours=ordered_neighbours(ordered, m=m, near=near),
    order=order,
    n0=n0,
  )


@dataclass(frozen=True)
class MeanBasis:
  """The mean function's basis, and the survey extent it was built on.

  §3.2 specifies ``phi(q) = (1, n, e)`` -- a plane through the survey, meant to
  absorb regional depth and slope so the GP models only what is left. Measured
  on this seabed, it absorbs **13%** of the held-out variance. The GP is then
  asked to carry the other 87%, which it cannot do beyond its own correlation
  length: inside a gap wider than the kernel's reach the posterior returns the
  mean, and a plane that explains nothing is a bad answer. That is precisely
  where the map lost to averaging four neighbours.

  Held-out residual variance on the four-heading survey, 8 m blocked holdout,
  total variance 105.19:

  ==========================  ====  ==============
  basis                       ``p``  held-out var
  ==========================  ====  ==============
  constant                       1  105.19
  linear -- §3.2's              3   91.84
  quadratic                      6   53.89
  cubic                         10   26.02
  tensor B-spline 8x8          100   11.16
  **tensor B-spline 12x12**    196   **8.10**
  tensor B-spline 16x16        324    8.58
  tensor B-spline 24x24        676   15.07
  ==========================  ====  ==============

  The overfitting turn is visible and unambiguous, so 12 knots an axis is
  measured rather than guessed. A spline basis is also the *right shape* for
  the job: seabed depth is a smooth surface with no reason to be polynomial,
  and a high-degree polynomial misbehaves exactly at the survey edges where the
  map is least constrained.

  :param kind: ``"polynomial"`` or ``"spline"``.
  :param degree: Total degree for a polynomial, or the B-spline's degree.
  :param knots: Knots per axis. Splines only.
  :param lower: Per-axis minimum of the fitting extent. Splines only.
  :param upper: Per-axis maximum of the fitting extent. Splines only.

  Note:
      **A spline basis is not a pure function of position**, which is why this
      is an object rather than a function. Its knots are pinned to the survey's
      extent, so prediction must use the ones the fit used; recomputing them
      from a batch of query points would silently evaluate a different basis.
      The fitted map therefore carries this, and prediction goes through it.

  Note:
      Queries outside the fitted extent are **clamped** to it, so the mean goes
      flat rather than diverging. A polynomial mean has no such guard, which is
      the other reason to prefer splines here: a quartic surface extrapolated a
      few metres past the last sounding can return any depth at all.
  """

  kind: str = "polynomial"
  degree: int = 1
  knots: int = 12
  lower: tuple[float, ...] | None = None
  upper: tuple[float, ...] | None = None
  keep: tuple[int, ...] | None = None
  intercept: bool = False

  @classmethod
  def build(
    cls,
    points: ArrayLike,
    kind: str = "linear",
    degree: int | None = None,
    knots: int = 12,
    min_support: float = 5.0,
  ) -> MeanBasis:
    """Choose a basis and pin it to the extent of ``points``.

    :param points: The survey the mean is fitted on, shape ``(N, 2)``.
    :param kind: ``"linear"``, ``"quadratic"``, ``"cubic"``, ``"polynomial"``
        or ``"spline"``.
    :param degree: Overrides the degree implied by ``kind``.
    :param knots: Knots per axis, for ``"spline"``.
    :param min_support: Drop spline basis functions supported by fewer than
        this many points. See the note.

    Note:
        **Unsupported basis functions are dropped, and they must be.** A
        tensor-product grid is laid over the survey's bounding box, but a
        survey does not fill its bounding box -- four headings across this one
        cover a diamond, leaving the corners empty. A basis function over an
        empty corner is an all-zero column of ``H``, so ``H' K^-1 H`` is
        singular and REML's ``-1/2 log|H' K^-1 H|`` is undefined. Measured
        here: of 196 functions at 12 knots an axis, **25 have no data under
        them at all** and ``H`` has rank 171.

        Because B-splines are a partition of unity, a column's sum over the
        survey *is* the effective number of soundings supporting it, so
        ``min_support`` is a count rather than a tuned threshold. Keeping a
        column supported by a point or two is worse than dropping it: its
        coefficient is barely determined and the mean can swing wildly there,
        which is precisely where a map should be admitting ignorance instead.

    Note:
        **A pruned spline basis therefore carries an explicit intercept**,
        which a complete tensor-product basis neither needs nor tolerates --
        adding one to a partition of unity duplicates the sum of the others
        and leaves ``H`` rank deficient. Dropping columns destroys
        the partition of unity, so the retained functions no longer sum to one
        and the mean decays toward *zero* away from the survey -- on this
        seabed, 0 m against a true mean depth of -65.3 m, a 65 m error at any
        query more than 60 m from a sounding. With the intercept the same query
        returns -63.7 m, and the fit is no worse for it (training residual
        variance 7.65 against 7.70). The splines then model deviation from a
        constant depth rather than depth itself, which is the more natural
        reading of them anyway.
    """
    points = np.asarray(points, dtype=float)
    named = {"linear": 1, "quadratic": 2, "cubic": 3}

    if kind in named:
      return cls(kind="polynomial", degree=degree or named[kind])
    if kind == "polynomial":
      return cls(kind="polynomial", degree=degree or 1)
    if kind != "spline":
      raise ValueError(
        f"kind must be one of {sorted(named) + ['polynomial', 'spline']}, "
        f"got {kind!r}"
      )

    if knots < 2:
      raise ValueError(f"expected at least 2 knots, got {knots}")

    basis = cls(
      kind="spline",
      degree=3 if degree is None else degree,
      knots=knots,
      lower=tuple(points.min(axis=0)),
      upper=tuple(points.max(axis=0)),
    )

    # On the raw tensor product: the assembled basis carries an intercept,
    # which would shift every index by one.
    support = basis._tensor(points).sum(axis=0)
    keep = np.nonzero(support >= min_support)[0]
    if len(keep) == 0:
      raise ValueError(
        f"no spline basis function is supported by {min_support} soundings; "
        f"the survey may be far smaller than {knots} knots an axis implies"
      )

    # Only a *pruned* basis needs the intercept. A complete tensor product is
    # already a partition of unity, so adding one would duplicate the sum of
    # the others exactly and leave H rank deficient.
    return replace(
      basis,
      keep=tuple(int(index) for index in keep),
      intercept=len(keep) < len(support),
    )

  @property
  def size(self) -> int:
    """Number of basis functions actually used, ``p``."""
    if self.kind == "polynomial":
      return (self.degree + 1) * (self.degree + 2) // 2
    kept = (
      (self.knots + self.degree - 1) ** 2
      if self.keep is None
      else len(self.keep)
    )
    return kept + (1 if self.intercept else 0)

  def _knot_vector(self, axis: int) -> np.ndarray:
    assert self.lower is not None and self.upper is not None
    low, high = self.lower[axis], self.upper[axis]
    interior = np.linspace(low, high, self.knots)
    pad = self.degree
    return np.concatenate([np.full(pad, low), interior, np.full(pad, high)])

  def __call__(self, points: ArrayLike) -> np.ndarray:
    """Evaluate the basis.

    :param points: Positions, shape ``(N, 2)``.
    :return: Shape ``(N, p)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
      raise ValueError(f"expected (N, 2) points, got {points.shape}")

    if self.kind == "polynomial":
      columns = [np.ones(len(points))]
      for total in range(1, self.degree + 1):
        for power in range(total + 1):
          columns.append(
            points[:, 0] ** (total - power) * points[:, 1] ** power
          )
      return np.column_stack(columns)

    values = self._tensor(points)
    if self.keep is not None:
      values = values[:, self.keep]
    if not self.intercept:
      return values

    # See the note on :meth:`build`.
    return np.column_stack([np.ones(len(points)), values])

  def _tensor(self, points: np.ndarray) -> np.ndarray:
    """The full tensor-product B-spline basis, before selection or intercept."""
    from scipy.interpolate import BSpline

    assert self.lower is not None and self.upper is not None
    axes = []
    for axis in (0, 1):
      clamped = np.clip(points[:, axis], self.lower[axis], self.upper[axis])
      axes.append(
        BSpline.design_matrix(
          clamped, self._knot_vector(axis), self.degree
        ).toarray()
      )

    north, east = axes
    return (north[:, :, None] * east[:, None, :]).reshape(len(points), -1)

  def gradient(self, points: ArrayLike) -> np.ndarray:
    """Derivative of the basis with respect to position.

    :param points: Positions, shape ``(N, 2)``.
    :return: Shape ``(N, p, 2)``, basis value per metre.

    Note:
        Needed because the map's mean gradient is ``d(phi)/dq . beta``, and the
        terrain update carries the sonar's range noise through it. For the
        linear basis this is the constant ``(0, 1, 0), (0, 0, 1)`` and the
        whole thing collapses to ``beta[1:]`` -- which is what the map used to
        assume, and which is silently wrong for any richer mean.

        Outside the fitted extent a spline basis is clamped, hence constant,
        hence flat: the gradient is zero there rather than whatever the last
        interior slope happened to be.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
      raise ValueError(f"expected (N, 2) points, got {points.shape}")

    if self.kind == "polynomial":
      columns = [np.zeros((len(points), 2))]
      for total in range(1, self.degree + 1):
        for power in range(total + 1):
          north_power, east_power = total - power, power
          north = points[:, 0] ** north_power
          east = points[:, 1] ** east_power
          d_north = (
            north_power * points[:, 0] ** (north_power - 1) * east
            if north_power
            else np.zeros(len(points))
          )
          d_east = (
            east_power * points[:, 1] ** (east_power - 1) * north
            if east_power
            else np.zeros(len(points))
          )
          columns.append(np.column_stack([d_north, d_east]))
      return np.stack(columns, axis=1)

    from scipy.interpolate import BSpline

    assert self.lower is not None and self.upper is not None
    values, slopes, inside = [], [], []
    for axis in (0, 1):
      raw = points[:, axis]
      clamped = np.clip(raw, self.lower[axis], self.upper[axis])
      knots = self._knot_vector(axis)
      width = self.knots + self.degree - 1

      spline = BSpline(knots, np.eye(width), self.degree)
      values.append(np.asarray(spline(clamped)))
      slopes.append(np.asarray(spline.derivative()(clamped)))
      inside.append(raw == clamped)

    north, east = values
    d_north, d_east = slopes

    # A clamped axis contributes no slope along itself.
    d_north = d_north * inside[0][:, None]
    d_east = d_east * inside[1][:, None]

    by_north = (d_north[:, :, None] * east[:, None, :]).reshape(len(points), -1)
    by_east = (north[:, :, None] * d_east[:, None, :]).reshape(len(points), -1)
    slope = np.stack([by_north, by_east], axis=-1)
    if self.keep is not None:
      slope = slope[:, self.keep]
    if not self.intercept:
      return slope

    # The intercept is constant, so it contributes no slope.
    return np.concatenate([np.zeros((len(points), 1, 2)), slope], axis=1)


#: §3.2's mean, kept as the default so nothing changes without being asked for.
LINEAR_MEAN = MeanBasis(kind="polynomial", degree=1)


def design_matrix(
  points: ArrayLike, basis: MeanBasis | None = None
) -> np.ndarray:
  """Evaluate a mean basis, defaulting to §3.2's ``phi(q) = (1, n, e)``.

  :param points: Positions, shape ``(N, 2)``.
  :param basis: The basis to use. Defaults to the linear one.
  :return: Shape ``(N, p)``.
  """
  return (basis or LINEAR_MEAN)(points)


def _gather_block(values: Tensor, neighbours: Tensor, rows: Tensor) -> Tensor:
  """Gather a per-sounding quantity over ``(g(i), i)`` -- neighbours, self last.

  Putting the sounding itself in the final row is what lets one Cholesky yield
  both the conditional mean and the conditional variance: for ``L = chol(K)``
  over ``(g(i), i)``, the last row holds ``L_gg^-1 K_gi`` and the conditional
  standard deviation, so no second solve is needed. GraphGP's ``refine``
  arrives at the same arrangement independently.

  :param values: Per-sounding values, shape ``(N,)`` or ``(N, d)``.
  :param neighbours: Conditioning indices for these rows, shape ``(b, m)``.
  :param rows: The soundings themselves, shape ``(b,)``.
  :return: Shape ``(b, m+1)`` or ``(b, m+1, d)``.
  """
  return torch.cat([values[neighbours], values[rows][:, None]], dim=1)


def _cholesky(blocks: Tensor, where: str) -> Tensor:
  """Batched Cholesky that reports *which* block failed.

  ``torch.linalg.cholesky_ex`` returns a status rather than raising, which is
  what this package needs: ``pyproject.toml`` promotes ``RuntimeWarning`` to an
  error, so a silent NaN would surface later as an unrelated test failure
  rather than here.
  """
  factor, info = torch.linalg.cholesky_ex(blocks)

  if torch.any(info != 0):
    bad = torch.nonzero(info).flatten()
    raise ValueError(
      f"{where}: kernel block not positive definite at "
      f"{bad.numel()} of {len(blocks)} soundings (first: index "
      f"{int(bad[0])}). Soundings may be closer together than the kernel can "
      f"resolve; raise the jitter or decimate more coarsely."
    )

  return factor


def _gram_chunk(
  block_points: Tensor,
  block_values: Tensor,
  block_noise: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  floor: Tensor,
) -> tuple[Tensor, Tensor]:
  """One chunk's contribution to ``(U^T V)^T (U^T V)`` and ``sum log U_ii``.

  Kept as its own function so it can be wrapped in gradient checkpointing: the
  ``(b, m+1, m+1)`` factors are the whole memory cost, and they are cheaper to
  recompute in the backward pass than to keep.
  """
  width = block_values.shape[1] - 1

  kernel = matern52(block_points, block_points, log_amplitude, log_lengthscale)
  kernel = kernel + torch.diag_embed(block_noise + floor)

  factor = _cholesky(kernel, "conditioning block")
  solved = torch.linalg.solve_triangular(factor, block_values, upper=False)

  whitened = solved[:, width, :]
  return (
    whitened.T @ whitened,
    torch.log(factor[:, width, width]).sum(),
  )


def whiten_gram(
  structure: VecchiaStructure,
  values: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
) -> tuple[Tensor, Tensor]:
  """``(U^T V)^T (U^T V)`` and ``sum_i log U_ii``, in bounded memory.

  Both objectives need only **cross-products** of the whitened columns, never
  the whitened columns themselves: the plain likelihood wants ``||U^T r||^2``,
  and REML wants ``W'W``, ``W'u`` and ``u'u``, which stacking ``[H | z]``
  delivers as one ``(p+1) x (p+1)`` gram. Every one of those is a sum over
  soundings, so they accumulate chunk by chunk.

  :param structure: Ordering and conditioning sets.
  :param values: Shape ``(N, p)``, in the structure's order.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Soundings per factorisation.
  :return: ``(gram, sum_i log U_ii)`` of shapes ``(p, p)`` and ``()``.

  Note:
      **Accumulating the gram is not by itself enough to bound memory**, and
      assuming otherwise cost a twenty-minute fit. Chunking bounds the working
      set, but autograd keeps every chunk's ``(b, m+1, m+1)`` factor alive for
      the backward pass, so peak memory stays proportional to ``N``. At the
      real survey's 424k training soundings that is 3.3 GB per intermediate and
      over 22 GB in total -- an out-of-memory error on a 24 GB card.

      The per-chunk ``backward()`` this plan originally called for does not
      rescue it either: REML's loss does not decompose over chunks, because
      ``beta`` depends on the whole of ``H' K^-1 H``.

      What does work is **gradient checkpointing**: store only each chunk's
      inputs and outputs and recompute its interior during the backward pass.
      Memory then follows ``chunk`` rather than ``N``, at the cost of one extra
      forward. That is the right trade here -- the forward is a batched
      Cholesky the GPU does in milliseconds, and the alternative is not fitting
      at all.
  """
  device, dtype = values.device, values.dtype
  points = torch.as_tensor(structure.points, dtype=dtype, device=device)
  noise = torch.as_tensor(noise, dtype=dtype, device=device).expand(
    structure.size
  )
  floor = jitter * torch.exp(log_amplitude)

  n0 = structure.n0
  head_points = points[:n0]
  head = matern52(head_points, head_points, log_amplitude, log_lengthscale)
  head = head + torch.diag(noise[:n0] + floor)

  factor = _cholesky(head[None], "head block")[0]
  solved = torch.linalg.solve_triangular(factor, values[:n0], upper=False)

  gram = solved.T @ solved
  log_diagonal = torch.log(torch.diagonal(factor)).sum()

  if n0 < structure.size:
    neighbours = torch.as_tensor(
      structure.neighbours[n0:], dtype=torch.long, device=device
    )
    tail = torch.arange(n0, structure.size, device=device)

    for start in range(0, len(tail), chunk):
      rows = tail[start : start + chunk]
      conditioning = neighbours[start : start + chunk]

      piece, diagonal = checkpoint(
        _gram_chunk,
        _gather_block(points, conditioning, rows),
        _gather_block(values, conditioning, rows),
        _gather_block(noise, conditioning, rows),
        log_amplitude,
        log_lengthscale,
        floor,
        use_reentrant=False,
      )
      gram = gram + piece
      log_diagonal = log_diagonal + diagonal

  return gram, -log_diagonal


def whiten(
  structure: VecchiaStructure,
  values: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
) -> tuple[Tensor, Tensor]:
  """Apply ``U^T`` to one or more vectors, and report ``sum_i log U_ii``.

  ``U`` is the sparse upper-triangular factor of the approximated precision,
  ``K^-1 = U U^T``. It is never assembled: the ``i``-th row of ``U^T v`` is the
  last entry of ``L_i^-1 v[c_i]`` for the block factor ``L_i``, and the head
  block contributes its whole dense solve. So whitening is exactly the work the
  likelihood already does, and applying it to a matrix rather than a vector
  costs only a wider triangular solve against Cholesky factors that are shared.

  That is what makes REML affordable: every quantity it needs beyond the plain
  likelihood is a product against this same ``U``.

  :param structure: Ordering and conditioning sets.
  :param values: Shape ``(N,)`` or ``(N, p)``, in the structure's order.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once. Memory only.
  :return: ``(U^T values, sum_i log U_ii)``, the first with the shape of
      ``values`` and the second a scalar.
  """
  flat = values.ndim == 1
  matrix = values[:, None] if flat else values

  device, dtype = matrix.device, matrix.dtype
  points = torch.as_tensor(structure.points, dtype=dtype, device=device)
  noise = torch.as_tensor(noise, dtype=dtype, device=device).expand(
    structure.size
  )
  floor = jitter * torch.exp(log_amplitude)

  n0 = structure.n0
  head_points = points[:n0]
  head = matern52(head_points, head_points, log_amplitude, log_lengthscale)
  head = head + torch.diag(noise[:n0] + floor)

  factor = _cholesky(head[None], "head block")[0]
  pieces = [torch.linalg.solve_triangular(factor, matrix[:n0], upper=False)]
  log_diagonal = torch.log(torch.diagonal(factor)).sum()

  if n0 < structure.size:
    neighbours = torch.as_tensor(
      structure.neighbours[n0:], dtype=torch.long, device=device
    )
    tail = torch.arange(n0, structure.size, device=device)
    width = structure.conditioning

    for start in range(0, len(tail), chunk):
      rows = tail[start : start + chunk]
      conditioning = neighbours[start : start + chunk]

      block_points = _gather_block(points, conditioning, rows)
      block_values = _gather_block(matrix, conditioning, rows)
      block_noise = _gather_block(noise, conditioning, rows)

      kernel = matern52(
        block_points, block_points, log_amplitude, log_lengthscale
      )
      kernel = kernel + torch.diag_embed(block_noise + floor)

      factor = _cholesky(kernel, "conditioning block")
      solved = torch.linalg.solve_triangular(factor, block_values, upper=False)

      pieces.append(solved[:, width, :])
      log_diagonal = log_diagonal + torch.log(factor[:, width, width]).sum()

  whitened = torch.cat(pieces, dim=0)
  # U_ii is the reciprocal of the conditional standard deviation, so the sum of
  # its logs is the negation of what the factors carry.
  return (whitened[:, 0] if flat else whitened), -log_diagonal


def sparse_factor(
  structure: VecchiaStructure,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
):
  """Assemble ``U`` explicitly, the sparse factor with ``K^-1 = U U^T``.

  The likelihood never needs this -- :func:`whiten` applies ``U^T`` without
  forming it, which is the whole reason the fit is cheap. Drawing from the
  model needs the opposite operation, ``solve(U^T, z)``, and a triangular solve
  does need the entries.

  :param structure: Ordering and conditioning sets.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once. Memory only.
  :return: ``U`` as a ``scipy.sparse`` CSC matrix, shape ``(N, N)``, upper
      triangular, in the **structure's** order.

  Note:
      Column ``i`` holds the coefficients of the ``i``-th conditional. Writing
      ``w`` for the regression of sounding ``i`` on its conditioning set and
      ``d`` for the conditional standard deviation, ``U[i, i] = 1/d`` and
      ``U[g(i), i] = -w/d``, because the whitened residual is
      ``(x_i - w' x_g) / d``. Both fall out of the block Cholesky that
      :func:`whiten` already computes: with ``L`` over ``(g(i), i)``,
      ``w = solve(L_gg^T, L[m, :m])`` and ``d = L[m, m]``.

      The head block is the same statement for a dense exact factorisation:
      ``K = L L^T`` gives ``K^-1 = L^-T L^-1``, so its corner of ``U`` is
      ``L^-T``.
  """
  from scipy.sparse import csc_matrix

  device, dtype = log_amplitude.device, log_amplitude.dtype
  points = torch.as_tensor(structure.points, dtype=dtype, device=device)
  noise = torch.as_tensor(noise, dtype=dtype, device=device).expand(
    structure.size
  )
  floor = jitter * torch.exp(log_amplitude)

  size, n0 = structure.size, structure.n0
  rows: list[np.ndarray] = []
  columns: list[np.ndarray] = []
  values: list[np.ndarray] = []

  head_points = points[:n0]
  head = matern52(head_points, head_points, log_amplitude, log_lengthscale)
  head = head + torch.diag(noise[:n0] + floor)
  factor = _cholesky(head[None], "head block")[0]

  # L^-T, by solving L^T X = I.
  corner = torch.linalg.solve_triangular(
    factor.T, torch.eye(n0, dtype=dtype, device=device), upper=True
  )
  corner = torch.triu(corner).cpu().numpy()

  nonzero = np.nonzero(corner)
  rows.append(nonzero[0])
  columns.append(nonzero[1])
  values.append(corner[nonzero])

  if n0 < size:
    neighbours = torch.as_tensor(
      structure.neighbours[n0:], dtype=torch.long, device=device
    )
    tail = torch.arange(n0, size, device=device)
    width = structure.conditioning

    for start in range(0, len(tail), chunk):
      block_rows = tail[start : start + chunk]
      conditioning = neighbours[start : start + chunk]

      block_points = _gather_block(points, conditioning, block_rows)
      block_noise = _gather_block(noise, conditioning, block_rows)

      kernel = matern52(
        block_points, block_points, log_amplitude, log_lengthscale
      )
      kernel = kernel + torch.diag_embed(block_noise + floor)
      factor = _cholesky(kernel, "conditioning block")

      lower = factor[:, :width, :width]
      cross = factor[:, width, :width].unsqueeze(-1)
      deviation = factor[:, width, width]

      weights = torch.linalg.solve_triangular(
        lower.transpose(-1, -2), cross, upper=True
      ).squeeze(-1)

      reciprocal = 1.0 / deviation
      off = (-weights * reciprocal[:, None]).cpu().numpy()

      index = conditioning.cpu().numpy()
      here = block_rows.cpu().numpy()

      rows.append(index.ravel())
      columns.append(np.repeat(here, width))
      values.append(off.ravel())

      rows.append(here)
      columns.append(here)
      values.append(reciprocal.cpu().numpy())

  row = np.concatenate(rows)
  column = np.concatenate(columns)
  value = np.concatenate(values)

  # Rows below n0 carry -1 padding in `neighbours`; those entries are not part
  # of any conditioning set and must not reach the matrix.
  keep = row >= 0
  return csc_matrix(
    (value[keep], (row[keep], column[keep])), shape=(size, size)
  )


def draw(
  structure: VecchiaStructure,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  count: int = 1,
  seed: int | None = None,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
) -> np.ndarray:
  """Draw realisations of the field at the structure's points.

  :param structure: Ordering and conditioning sets.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param count: How many independent realisations.
  :param seed: Seed for the white noise.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once. Memory only.
  :return: Shape ``(N,)`` when ``count`` is 1, else ``(N, count)``, in the
      **caller's** order -- the inverse of ``structure.order``, so it lines up
      with whatever positions were handed to :func:`build_structure`.

  Note:
      ``x = U^-T z`` for white ``z`` has covariance ``U^-T U^-1 = K``, so one
      sparse triangular solve is the whole draw.

  Note:
      **What this is for, and the trap in it.** A draw from a field whose
      hyperparameters are known is the only way to ask whether a fit *recovers*
      them, as opposed to merely converging. But generating and fitting with
      the same conditioning set is circular -- the data is then an exact sample
      from the model being fitted, and recovery tests the optimiser rather than
      the approximation. Generate with a conditioning set several times larger
      than the one being tested.
  """
  from scipy.sparse.linalg import spsolve_triangular

  factor = sparse_factor(
    structure,
    log_amplitude,
    log_lengthscale,
    noise,
    jitter=jitter,
    chunk=chunk,
  )

  generator = np.random.default_rng(seed)
  white = generator.standard_normal((structure.size, count))

  ordered = spsolve_triangular(factor.T.tocsr(), white, lower=True)

  result = np.empty_like(ordered)
  result[structure.order] = ordered
  return result[:, 0] if count == 1 else result


def vecchia_loglik(
  structure: VecchiaStructure,
  residual: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
) -> Tensor:
  """Log marginal likelihood of a residual whose mean is already fixed.

  ``log p = -N/2 log 2pi + sum_i log U_ii - 1/2 ||U^T r||^2``, both terms from
  :func:`whiten`.

  This is maximum likelihood with a **plug-in** mean. Prefer
  :func:`vecchia_reml` unless the plug-in is what you specifically want --
  estimating the mean from the data shortens the residual, and this objective
  reads that as a smaller amplitude.

  :param structure: Ordering and conditioning sets.
  :param residual: ``z - phi(q)^T beta`` in the structure's order, shape ``(N,)``.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once.
  :return: Scalar, differentiable in the hyperparameters and in ``noise``.
  """
  gram, log_determinant = whiten_gram(
    structure,
    residual[:, None],
    log_amplitude,
    log_lengthscale,
    noise,
    jitter=jitter,
    chunk=chunk,
  )

  return -0.5 * structure.size * _LOG_2PI + log_determinant - 0.5 * gram[0, 0]


def vecchia_reml(
  structure: VecchiaStructure,
  depth: Tensor,
  basis: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
  noise: Tensor,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
) -> tuple[Tensor, Tensor]:
  """Restricted log likelihood, and the generalised-least-squares mean.

  ``ell_R = -(N-p)/2 log 2pi + sum_i log U_ii - 1/2 log|H' K^-1 H|
            - 1/2 (z - H beta)' K^-1 (z - H beta)``

  **Why not plain maximum likelihood.** Estimating ``beta`` from the data
  projects variance out of the residual at a rate ``tr(HK)/tr(K)`` -- measured
  here at 8-13% for a 12 m lengthscale over an 80 m box -- and a plug-in
  likelihood reads that shortfall as a smaller amplitude. Note the culprit is
  the *projection*, not the estimator: ordinary least squares is unbiased for
  ``beta``, and switching it for generalised least squares changes nothing,
  measured at -10.6% against OLS's -9.7%. REML instead scores the error
  contrasts orthogonal to ``H``, which is what the extra log-determinant term
  is, and comes in at -3.0% on the same problem.

  That direction matters here: an amplitude fitted low understates the map's
  own uncertainty, which is what the smoother's update would then be too
  confident about.

  **Why it is cheap.** Everything beyond the plain likelihood is a product
  against the same ``U``: ``H' K^-1 H = (U'H)' (U'H)`` and
  ``beta = (H' K^-1 H)^-1 (U'H)' (U'z)``. So ``H`` and ``z`` whiten together in
  one pass, the Cholesky factors are shared, and what is left is a ``3 x 3``
  solve and log-determinant. A few per cent per evaluation.

  :param structure: Ordering and conditioning sets.
  :param depth: Seabed elevation in the structure's order, shape ``(N,)``.
  :param basis: The linear mean's design matrix ``phi(q)``, shape ``(N, p)``,
      in the same order.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(2,)``.
  :param noise: Per-sounding variance, shape ``(N,)``; a scalar broadcasts.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once.
  :return: ``(restricted log likelihood, beta)``.

  Note:
      ``beta`` is recomputed here at **every** hyperparameter value, from the
      current kernel. Holding it fixed from an earlier fit would make this
      maximum likelihood with a plug-in mean again, which is the thing the
      objective exists to avoid.

  Note:
      Because ``K^-1`` is itself the Vecchia approximation, this is
      Vecchia-REML -- the same thing GpGp and GPvecchia do for covariates. It
      is consistent with the rest of the pipeline, so there is no exactness
      mismatch between the objective and the map it fits.

  Note:
      The constant ``+1/2 log|H'H|`` that makes REML invariant to the
      parametrisation of ``H`` is omitted: it does not depend on the
      hyperparameters, so it shifts the objective without moving its maximum.
      Restricted likelihoods are comparable across kernels, **not** across
      different designs.
  """
  stacked = torch.cat([basis, depth[:, None]], dim=1)
  gram, log_determinant = whiten_gram(
    structure,
    stacked,
    log_amplitude,
    log_lengthscale,
    noise,
    jitter=jitter,
    chunk=chunk,
  )

  # The stacked gram holds every cross-product the objective needs:
  # H'K^-1 H, H'K^-1 z and z'K^-1 z, in one (p+1) x (p+1) block.
  information = gram[:-1, :-1]
  cross = gram[:-1, -1]
  beta = torch.linalg.solve(information, cross)

  # ||u - W beta||^2 = u'u - b' A^-1 b, without ever forming u or W.
  quadratic = gram[-1, -1] - cross @ beta

  restricted = (
    -0.5 * (structure.size - basis.shape[1]) * _LOG_2PI
    + log_determinant
    - 0.5 * torch.logdet(information)
    - 0.5 * quadratic
  )

  return restricted, beta


@dataclass(frozen=True)
class VecchiaHyperparameters:
  """Kernel parameters, held in logs because that is how they are fitted.

  Log-parameterised so the optimiser cannot step them negative and so a step
  means the same proportional change wherever it starts -- an amplitude of
  0.01 and one of 100 both move by the same factor per step.

  :param log_amplitude: ``log(sigma_f^2)``, the marginal variance of the
      seabed about its linear trend, in metres squared.
  :param log_lengthscale: ``log`` of the per-axis correlation length, metres.
  :param log_noise: ``log(sigma_z^2)``, the base per-sounding variance.
  """

  log_amplitude: float
  log_lengthscale: tuple[float, float]
  log_noise: float

  @property
  def amplitude(self) -> float:
    """``sigma_f^2``, metres squared."""
    return math.exp(self.log_amplitude)

  @property
  def lengthscale(self) -> np.ndarray:
    """Per-axis correlation length, **metres** -- readable as a distance."""
    return np.exp(np.asarray(self.log_lengthscale, dtype=float))

  @property
  def noise(self) -> float:
    """``sigma_z^2``, metres squared."""
    return math.exp(self.log_noise)


@dataclass
class VecchiaMap:
  """A fitted seabed map.

  Everything needed to answer a query, and nothing that has to be recomputed:
  the ordering and conditioning sets are in :attr:`structure`, the linear trend
  in :attr:`beta`, and what the GP models is :attr:`residual` -- the survey
  with that trend removed.

  :param structure: Ordering and conditioning sets.
  :param residual: ``z - phi(q)^T beta`` in the structure's order, shape ``(N,)``.
  :param beta: Linear-mean coefficients ``(intercept, d/dn, d/de)``.
  :param hyper: Fitted kernel parameters.
  :param noise: Per-sounding variance ``sigma_j^2``, shape ``(N,)``, in the
      structure's order. Equal to ``hyper.noise`` everywhere until the
      input-noise pass inflates it.
  :param loglik_trace: Log likelihood at each optimiser step, so convergence is
      observable rather than assumed.
  :param fit_device: Where the fit actually ran.
  :param information: ``H' K^-1 H``, shape ``(p, p)``. Kept from the fit
      because prediction needs it to propagate the uncertainty in ``beta``, and
      recomputing it would mean whitening the whole survey again.
  :param basis: The mean's basis. Carried on the map because a spline basis is
      pinned to the extent it was fitted on, so prediction must evaluate the
      same one; see :class:`MeanBasis`.
  """

  structure: VecchiaStructure
  residual: np.ndarray
  beta: np.ndarray
  hyper: VecchiaHyperparameters
  noise: np.ndarray
  loglik_trace: list[float]
  basis: MeanBasis = LINEAR_MEAN
  fit_device: str = "cpu"
  information: np.ndarray | None = None
  _tree: object | None = field(default=None, repr=False, compare=False)

  @property
  def lengthscale(self) -> np.ndarray:
    """Per-axis correlation length in metres. See :class:`VecchiaHyperparameters`."""
    return self.hyper.lengthscale

  @property
  def tree(self):
    """A ``KDTree`` over the survey, built **once** and kept.

    ``navigate.py`` queries the map every ping at 5 Hz. Rebuilding a tree over
    a hundred thousand soundings on each of those would dominate the tick by
    orders of magnitude, so it is built lazily here and reused. Lazily rather
    than in ``__init__`` so that loading a checkpoint stays cheap for callers
    that only want the hyperparameters.
    """
    if self._tree is None:
      from scipy.spatial import KDTree

      self._tree = KDTree(self.structure.points)
    return self._tree

  def predict_joint(
    self,
    points: ArrayLike,
    observation_noise: bool = True,
    jitter: float = DEFAULT_JITTER,
    beta_uncertainty: bool = True,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Joint Gaussian over the seabed at a set of horizontal positions.

    Implements :class:`~auv_pose.estimation.terrain.DepthMap`, which is what
    the smoother's update consumes. The queries condition on each other as well
    as on the survey, so the off-diagonals are real: they are what stop a fan of
    adjacent beams being counted as that many independent constraints.

    :param points: ``(..., b, 2)`` of ``(x, y)`` in metres. Leading axes are
        batch dimensions, so a whole sigma-point cloud is answered at once.
    :param observation_noise: Include ``sigma_z^2``, giving the spread of a
        *sounding* rather than of the seabed.
    :param jitter: Diagonal regulariser, relative to the amplitude.
    :param beta_uncertainty: Propagate the uncertainty in the linear mean.
        Leave on: ``beta`` was estimated, and omitting its contribution
        understates the map exactly where queries sit far from the survey.
    :return: ``(mean, cov)`` of shapes ``(..., b)`` and ``(..., b, b)``, metres
        and metres squared. Depth is **z-up**: a seabed 65 m down is ``-65``.

    Note:
        The queries are reordered internally -- maximin, predictions last --
        and put back before returning. For a multibeam fan that matters: in
        beam order every predecessor of a beam lies to one side of it, so
        conditioning tells it little about the other.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim < 2 or points.shape[-1] != 2:
      raise ValueError(f"expected (..., b, 2) points, got {points.shape}")

    leading = points.shape[:-2]
    beams = points.shape[-2]
    if beams > self.structure.conditioning:
      raise ValueError(
        f"{beams} query points against a conditioning set of "
        f"{self.structure.conditioning}: beyond m the queries stop "
        "conditioning on each other and the joint covariance loses structure. "
        "Decimate the fan or refit with a larger m."
      )

    mean, covariance = _predict_joint_array(
      self,
      points.reshape(-1, beams, 2),
      observation_noise=observation_noise,
      jitter=jitter,
      beta_uncertainty=beta_uncertainty,
    )

    return (
      mean.reshape(*leading, beams),
      covariance.reshape(*leading, beams, beams),
    )

  def mean_gradient(
    self,
    points: ArrayLike,
    chunk_size: int = 5000,
    jitter: float = DEFAULT_JITTER,
  ) -> np.ndarray:
    """Slope of the map's mean, in metres per metre.

    Implements :class:`~auv_pose.estimation.terrain.DepthMap`. Used to carry
    the sonar's range noise through the shift in *where* the map is queried,
    and, during fitting, to carry the survey's own positional uncertainty into
    the per-sounding noise.

    Analytic, by differentiating the kernel with respect to the query position.
    For a single query conditioning on its ``m`` nearest soundings the mean is
    ``phi(q)' beta + k(q, g) K_gg^-1 r_g``, so::

        grad mu(q) = [d phi(q)/dq]' beta + [dk(q, g)/dq]' K_gg^-1 r_g

    :param points: ``(n, 2)`` of ``(x, y)`` in metres.
    :param chunk_size: Points per batch, to bound memory.
    :param jitter: Diagonal regulariser, relative to the amplitude.
    :return: ``(n, 2)`` of ``d(elevation)/dx, d(elevation)/dy``.

    Note:
        **Per point**, matching the contract and matching :meth:`predict`: each
        query conditions on the survey alone, not on the other points in the
        call. :meth:`predict_joint` deliberately does condition queries on each
        other; this does not, because a gradient is a property of one location.

    Note:
        Unlike the conditional variance, this needs a full solve against the
        block rather than only the last row of its factor -- the whole of
        ``K_gg^-1 r_g`` appears, not just the standardised residual.

    Note:
        The mean is only **piecewise** smooth: the conditioning set changes
        discontinuously as a query crosses between soundings, so this analytic
        gradient and a central difference taken across such a boundary will
        disagree. Both are right about their own side of it. Comparisons
        against finite differences belong at full conditioning, where there is
        no boundary to cross.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[-1] != 2:
      raise ValueError(f"expected (n, 2) points, got {points.shape}")

    structure = self.structure
    width = min(structure.conditioning, structure.size)

    log_amplitude = torch.tensor(self.hyper.log_amplitude, dtype=torch.float64)
    log_lengthscale = torch.tensor(
      self.hyper.log_lengthscale, dtype=torch.float64
    )
    floor = jitter * self.hyper.amplitude

    survey = torch.as_tensor(structure.points, dtype=torch.float64)
    residual = torch.as_tensor(self.residual, dtype=torch.float64)
    noise = torch.as_tensor(self.noise, dtype=torch.float64)

    gradients: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
      block = points[start : start + chunk_size]
      _, neighbours = self.tree.query(block, k=width)
      neighbours = torch.as_tensor(
        np.atleast_2d(neighbours.reshape(len(block), width))
      )

      local = survey[neighbours]
      kernel = matern52(local, local, log_amplitude, log_lengthscale)
      kernel = kernel + torch.diag_embed(noise[neighbours] + floor)

      factor = _cholesky(kernel, "gradient block")
      weights = torch.cholesky_solve(residual[neighbours][..., None], factor)

      query = torch.as_tensor(block, dtype=torch.float64)[:, None, :]
      slope = matern52_gradient(
        query, local, log_amplitude, log_lengthscale
      ).squeeze(1)

      mean_slope = np.einsum("npd,p->nd", self.basis.gradient(block), self.beta)
      gradients.append((slope * weights).sum(dim=1).numpy() + mean_slope)

    return np.concatenate(gradients)

  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = 5000,
    with_std: bool = False,
    observation_noise: bool = False,
    jitter: float = DEFAULT_JITTER,
  ):
    """Predict seabed elevation at horizontal positions, one point at a time.

    Signature-compatible with :meth:`auv_pose.mapping.svgp.BathymetryMap.predict`
    so ``navigate.py`` and ``predict_depth.py`` need no change.

    :param points: ``(n, 2)`` of ``(x, y)`` in metres.
    :param chunk_size: Points per batch, to bound memory.
    :param with_std: Also return the posterior standard deviation, in metres.
    :param observation_noise: Add ``sigma_z^2``, giving the spread of a
        sounding rather than of the seabed.
    :param jitter: Diagonal regulariser, relative to the amplitude.
    :return: Elevations ``(n,)``, or ``(elevation, std)`` when ``with_std``.

    Note:
        This is the **per-point** prediction: each query conditions on the
        survey alone, not on the other points in the call. It therefore does
        not agree exactly with the diagonal of :meth:`predict_joint`, which
        conditions each query on those before it and so is a little tighter.
        Both are honest; they answer different questions, and this is the one
        a filter asks at tick rate.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[-1] != 2:
      raise ValueError(f"expected (n, 2) points, got {points.shape}")

    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []

    for start in range(0, len(points), chunk_size):
      block = points[start : start + chunk_size]
      mean, covariance = _predict_joint_array(
        self,
        block[:, None, :],
        observation_noise=observation_noise,
        jitter=jitter,
        beta_uncertainty=True,
      )
      means.append(mean[:, 0])
      variances.append(covariance[:, 0, 0])

    elevation = np.concatenate(means)
    if not with_std:
      return elevation
    return elevation, np.sqrt(np.concatenate(variances))


def initial_hyperparameters(
  points: ArrayLike, residual: ArrayLike, ard: bool = True
) -> VecchiaHyperparameters:
  """A starting point for the fit, taken from the data rather than guessed.

  With no input standardisation the optimiser sees parameters in metres, so
  where it starts matters more than it would otherwise -- a poor initialisation
  is the usual way a hand-rolled marginal-likelihood fit fails. These three
  lines carry the weight a scaler used to.

  :param points: Sounding positions, shape ``(N, 2)``.
  :param residual: Survey depths with the linear trend removed, shape ``(N,)``.
  :param ard: Per-axis lengthscales. When ``False`` both axes start together
      and the fit keeps them so.
  :return: Amplitude from the residual's variance, lengthscales from a tenth of
      the survey extent, and a nugget at one per cent of the amplitude.
  """
  points = np.asarray(points, dtype=float)
  residual = np.asarray(residual, dtype=float)

  extent = points.max(axis=0) - points.min(axis=0)
  lengthscale = np.maximum(extent / 10.0, 1e-3)
  if not ard:
    lengthscale = np.full(2, float(np.exp(np.log(lengthscale).mean())))

  amplitude = max(float(residual.var()), 1e-12)

  return VecchiaHyperparameters(
    log_amplitude=math.log(amplitude),
    log_lengthscale=(
      float(np.log(lengthscale[0])),
      float(np.log(lengthscale[1])),
    ),
    log_noise=math.log(0.01 * amplitude),
  )


def fit_vecchia(
  points: ArrayLike,
  depth: ArrayLike,
  m: int = 30,
  n0: int | None = None,
  noise_inflation: ArrayLike | None = None,
  steps: int = 300,
  learning_rate: float = 0.05,
  ard: bool = True,
  method: str = "reml",
  device: str | torch.device | None = None,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
  near: int | None = None,
  mean: str = "linear",
  mean_knots: int = 12,
  mean_support: float = 5.0,
) -> VecchiaMap:
  """Fit the linear mean and the kernel hyperparameters to a survey.

  Under the default ``method="reml"`` the mean is *profiled out* rather than
  fitted once up front: ``beta`` is the generalised-least-squares estimate
  recomputed at every hyperparameter value, and what is scored is the error
  contrasts orthogonal to the design. §3.2's two-stage description -- ``beta``
  by least squares, then hyperparameters by marginal likelihood -- is
  ``method="ml"``, kept for the comparison that motivated the change.

  :param points: Sounding positions, shape ``(N, 2)``, metres, any order.
  :param depth: Seabed elevation at each, shape ``(N,)``. **z-up**, so a seabed
      65 m down is ``-65``.
  :param m: Conditioning-set size.
  :param n0: Dense head-block size; see :func:`build_structure`.
  :param noise_inflation: Per-sounding variance to add to ``sigma_z^2``, shape
      ``(N,)``, in the **caller's** order. This is the input-noise term; zero
      unless a previous pass computed it.
  :param steps: Adam steps.
  :param learning_rate: Adam step size, on the log parameters.
  :param ard: Fit a lengthscale per axis. ``False`` ties them together.
  :param method: ``"reml"`` to score the error contrasts and take ``beta`` as
      the generalised-least-squares estimate at each step, or ``"ml"`` for
      maximum likelihood with ``beta`` fixed once by ordinary least squares.
      See :func:`vecchia_reml` for why the default is not ``"ml"``.
  :param device: Where to fit. Defaults to CUDA when available -- the cost is
      one batched Cholesky per step, which is where the GPU pays.
  :param jitter: Diagonal regulariser, relative to the amplitude.
  :param chunk: Blocks factorised at once.
  :param near: How many of the ``m`` conditioning points are nearest
      neighbours; the remainder are spread across the ordering. ``None``, the
      default, is all-nearest.

      Stein, Chi and Welty (2004) find all-nearest to be the *worst* design
      for estimating a range parameter under a linear mean -- see
      :func:`~auv_pose.mapping.ordering.ordered_neighbours` for their numbers
      -- so the default is expected to be the wrong one here. It stays until a
      conditioning ladder on real soundings says by how much, because their
      result is measured on a different field with a different ordering, and
      this map's whole difficulty has been numbers that did not transfer.
  :param mean: Mean basis -- ``"linear"`` (§3.2's), ``"quadratic"``,
      ``"cubic"`` or ``"spline"``. See :class:`MeanBasis` for what each absorbs
      on this seabed; the linear one absorbs 13%.
  :param mean_knots: Knots per axis when ``mean="spline"``.
  :param mean_support: Soundings a spline basis function needs before it is
      kept; see :meth:`MeanBasis.build`.
  :return: The fitted map.

  Note:
      Deterministic. There is no subsampling and no random initialisation, so
      the same survey and settings give the same map -- which matters because
      the ordering is stored in the checkpoint and a refit that disagreed with
      it would be a confusing thing to debug.

  Note:
      Under ``method="reml"`` -- the default -- ``beta`` is **not** a separate
      fitting step. It is the generalised-least-squares estimate under the
      current kernel, recomputed inside the objective at every hyperparameter
      value, and what comes back is the value at the optimum. §3.2 describes
      ``beta`` by least squares and the hyperparameters by marginal likelihood,
      as two stages; REML is one objective with ``beta`` profiled out, and the
      paper should say so.
  """
  points = np.asarray(points, dtype=float)
  depth = np.asarray(depth, dtype=float)
  if len(points) != len(depth):
    raise ValueError(f"{len(points)} positions against {len(depth)} depths")

  structure = build_structure(points, m=m, n0=n0, near=near)
  ordered_points = structure.points
  ordered_depth = depth[structure.order]

  if method not in ("reml", "ml"):
    raise ValueError(f'method must be "reml" or "ml", got {method!r}')

  mean_basis = MeanBasis.build(
    ordered_points, kind=mean, knots=mean_knots, min_support=mean_support
  )
  basis = design_matrix(ordered_points, mean_basis)
  # Only ever a starting point under REML, where the objective refits it.
  beta = np.linalg.lstsq(basis, ordered_depth, rcond=None)[0]
  residual = ordered_depth - basis @ beta

  start = initial_hyperparameters(ordered_points, residual, ard=ard)
  resolved = _resolve_device(device)

  inflation = (
    np.zeros(len(points))
    if noise_inflation is None
    else np.asarray(noise_inflation, dtype=float)[structure.order]
  )

  depth_tensor = torch.as_tensor(
    ordered_depth, dtype=torch.float64, device=resolved
  )
  basis_tensor = torch.as_tensor(basis, dtype=torch.float64, device=resolved)
  residual_tensor = torch.as_tensor(
    residual, dtype=torch.float64, device=resolved
  )
  inflation_tensor = torch.as_tensor(
    inflation, dtype=torch.float64, device=resolved
  )

  log_amplitude = torch.tensor(
    start.log_amplitude,
    dtype=torch.float64,
    device=resolved,
    requires_grad=True,
  )
  log_noise = torch.tensor(
    start.log_noise, dtype=torch.float64, device=resolved, requires_grad=True
  )
  # With ard off there is one lengthscale, broadcast to both axes, so the
  # optimiser cannot pull them apart.
  log_lengthscale = torch.tensor(
    list(start.log_lengthscale) if ard else [start.log_lengthscale[0]],
    dtype=torch.float64,
    device=resolved,
    requires_grad=True,
  )

  optimiser = torch.optim.Adam(
    [log_amplitude, log_lengthscale, log_noise], lr=learning_rate
  )

  trace: list[float] = []
  fitted_beta = torch.as_tensor(beta, dtype=torch.float64, device=resolved)

  for _ in range(steps):
    optimiser.zero_grad()
    lengthscale = log_lengthscale if ard else log_lengthscale.expand(2)
    variance = torch.exp(log_noise) + inflation_tensor

    if method == "reml":
      value, fitted_beta = vecchia_reml(
        structure,
        depth_tensor,
        basis_tensor,
        log_amplitude,
        lengthscale,
        variance,
        jitter=jitter,
        chunk=chunk,
      )
    else:
      value = vecchia_loglik(
        structure,
        residual_tensor,
        log_amplitude,
        lengthscale,
        variance,
        jitter=jitter,
        chunk=chunk,
      )

    (-value).backward()
    optimiser.step()
    trace.append(float(value.detach()))

  with torch.no_grad():
    final = log_lengthscale if ard else log_lengthscale.expand(2)
    if method == "reml":
      # One last evaluation, so beta matches the hyperparameters returned
      # rather than the ones from the step before the final update.
      _, fitted_beta = vecchia_reml(
        structure,
        depth_tensor,
        basis_tensor,
        log_amplitude,
        final,
        torch.exp(log_noise) + inflation_tensor,
        jitter=jitter,
        chunk=chunk,
      )
      beta = fitted_beta.cpu().numpy()
      residual = ordered_depth - basis @ beta

    gram, _ = whiten_gram(
      structure,
      basis_tensor,
      log_amplitude,
      final,
      torch.exp(log_noise) + inflation_tensor,
      jitter=jitter,
      chunk=chunk,
    )
    information = gram.cpu().numpy()

    fitted = VecchiaHyperparameters(
      log_amplitude=float(log_amplitude),
      log_lengthscale=(float(final[0]), float(final[1])),
      log_noise=float(log_noise),
    )

  return VecchiaMap(
    structure=structure,
    residual=residual,
    beta=beta,
    hyper=fitted,
    noise=fitted.noise + inflation,
    loglik_trace=trace,
    basis=mean_basis,
    fit_device=str(resolved),
    information=information,
  )


def fit_vecchia_nigp(
  points: ArrayLike,
  depth: ArrayLike,
  position_cov: ArrayLike,
  passes: int = 2,
  **kwargs,
) -> tuple[VecchiaMap, list[np.ndarray]]:
  """Fit with each sounding's position uncertainty carried into its noise.

  McHutchon and Rasmussen's noisy-input GP (NIGP), eq. 6: a position error
  ``e ~ N(0, Sigma_q)`` moves a sounding's depth by about ``grad(mu)' e``, so
  to first order it is extra *output* noise of variance
  ``s = grad(mu)' Sigma_q grad(mu)``. That is zero on flat seabed and large on
  a slope, which is the point: a misplaced sounding on a slope is a wrong
  depth, and one on the flat is harmless.

  The slope comes from the map being fitted, so this iterates: fit, take the
  mean's gradient at every sounding, refit with ``s`` frozen and ``sigma_z^2``
  still free. §3.2's procedure is two passes; the paper notes it can go on.

  :param points: Sounding positions, ``(N, 2)``.
  :param depth: Seabed elevation, ``(N,)``.
  :param position_cov: Each sounding's horizontal covariance, ``(N, 2, 2)``.
  :param passes: Fits to run, at least two: the first has no inflation, so a
      single pass would be a plain fit wearing NIGP's name.
  :param kwargs: Passed to every :func:`fit_vecchia`.
  :return: The last fit, and the inflation computed after each pass, so
      ``inflations[-1] - inflations[-2]`` shows whether it has settled.

  Note:
      This inflates the variance of a misplaced sounding; it does not move it
      back. A survey whose navigation drifted *consistently* -- the whole map
      shifted by the same error -- is biased, and no variance term fixes that.
  """
  if passes < 2:
    raise ValueError(
      f"NIGP needs at least two passes, got {passes}: the first has no "
      "inflation, so one pass is a plain fit"
    )
  points = np.asarray(points, dtype=float)
  position_cov = np.asarray(position_cov, dtype=float)
  if position_cov.shape != (len(points), 2, 2):
    raise ValueError(
      f"expected ({len(points)}, 2, 2) position covariance, "
      f"got {position_cov.shape}"
    )

  inflation = None
  inflations: list[np.ndarray] = []
  fitted = None
  for _ in range(passes):
    fitted = fit_vecchia(points, depth, noise_inflation=inflation, **kwargs)
    slope = fitted.mean_gradient(points)
    inflation = np.einsum("nd,nde,ne->n", slope, position_cov, slope)
    inflations.append(inflation)

  assert fitted is not None
  return fitted, inflations


def _query_order(queries: np.ndarray) -> np.ndarray:
  """Maximin order for each group of query positions, predictions last overall.

  Matters for a multibeam fan. In beam order the queries run along a line, so
  every predecessor of a beam sits on one side of it and conditioning on them
  says little about the other. Maximin spreads them, so each beam is bracketed.

  ``B`` is a few dozen, so the naive ``O(B^2)`` greedy is the right
  implementation and needs no tree.

  Note:
      Ordered **per group**, not once for the batch. Sharing one ordering was
      tempting -- the groups of a sigma-point cloud are near-identical in
      geometry, and any ordering gives a valid approximation -- but it makes
      the answer depend on how the caller batched the call. Measured at 0.06 m
      of disagreement in the mean between a batched and an unbatched query of
      the same points, which is not a difference a map should have.

  :param queries: Shape ``(G, B, 2)``.
  :return: Shape ``(G, B)``.
  """
  return np.stack([maximin_order(group) for group in queries])


def _conditioning_sets(
  survey_neighbours: np.ndarray,
  survey_distance: np.ndarray,
  queries: np.ndarray,
  index: int,
  width: int,
) -> tuple[np.ndarray, np.ndarray]:
  """The ``m`` nearest of (survey soundings, earlier queries) for one query.

  This is what makes the scheme RF-full rather than plain kriging: a query
  conditions on the queries ordered before it as well as on the survey, which
  is what puts the off-diagonal structure into ``cov_M``.

  :return: ``(indices, is_survey)``, each ``(G, m)``. Indices are into the
      survey where ``is_survey``, and into the ordered queries otherwise.
  """
  groups = len(queries)

  if index == 0:
    return survey_neighbours[:, index], np.ones((groups, width), dtype=bool)

  earlier = queries[:, :index]
  separation = np.linalg.norm(earlier - queries[:, index : index + 1], axis=-1)

  candidates = np.concatenate(
    [survey_neighbours[:, index], np.tile(np.arange(index), (groups, 1))],
    axis=1,
  )
  distance = np.concatenate([survey_distance[:, index], separation], axis=1)
  from_survey = np.concatenate(
    [
      np.ones((groups, width), dtype=bool),
      np.zeros((groups, index), dtype=bool),
    ],
    axis=1,
  )

  keep = np.argsort(distance, axis=1, kind="stable")[:, :width]
  return (
    np.take_along_axis(candidates, keep, axis=1),
    np.take_along_axis(from_survey, keep, axis=1),
  )


def _sparse_columns(
  fitted: VecchiaMap,
  queries: np.ndarray,
  jitter: float,
) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
  """The nonzero entries of ``U``'s query columns, one block at a time.

  Under the response-first ordering of Katzfuss et al. (2020) the joint is
  ``x = (z_o, y_p)`` and ``U`` is block upper triangular, so the factor the
  conditional needs is ``V = U_pp`` -- a *submatrix* of ``U``, not something to
  be recomputed. Its entries are filled here directly from each query's own
  block factor. Forming ``W = U U^T`` and factorising that instead would be
  both wasteful and numerically worse; the paper warns of the spurious
  nonzeros it produces.

  :return: ``(V, survey_weight, survey_index, order)`` -- the ``(G, B, B)``
      upper-triangular factor, the ``(G, B, m)`` entries of ``U`` at survey
      rows, the survey indices they belong to, and the query order applied.
  """
  structure = fitted.structure
  width = structure.conditioning
  groups, beams = queries.shape[:2]

  order = _query_order(queries)
  ordered = np.take_along_axis(queries, order[:, :, None], axis=1)

  if width > structure.size:
    raise ValueError(
      f"conditioning set of {width} exceeds the {structure.size} soundings "
      "in the survey; every query block would be ragged. Refit with a "
      "smaller m."
    )

  survey_distance, survey_neighbours = fitted.tree.query(
    ordered.reshape(-1, 2), k=width
  )
  survey_distance = survey_distance.reshape(groups, beams, width)
  survey_neighbours = survey_neighbours.reshape(groups, beams, width)

  log_amplitude = torch.tensor(fitted.hyper.log_amplitude, dtype=torch.float64)
  log_lengthscale = torch.tensor(
    fitted.hyper.log_lengthscale, dtype=torch.float64
  )
  floor = jitter * math.exp(fitted.hyper.log_amplitude)

  block_points = np.empty((groups, beams, width + 1, 2))
  block_noise = np.zeros((groups, beams, width + 1))
  indices = np.empty((groups, beams, width), dtype=np.int64)
  from_survey = np.empty((groups, beams, width), dtype=bool)

  for beam in range(beams):
    picked, survey = _conditioning_sets(
      survey_neighbours, survey_distance, ordered, beam, width
    )
    indices[:, beam] = picked
    from_survey[:, beam] = survey

    rows = np.arange(groups)[:, None]
    position = np.where(
      survey[..., None],
      structure.points[np.clip(picked, 0, structure.size - 1)],
      ordered[rows, np.clip(picked, 0, beams - 1)],
    )
    block_points[:, beam, :width] = position
    block_points[:, beam, width] = ordered[:, beam]

    # RF-full conditions on the latent wherever it can: a survey neighbour is a
    # response and carries its sounding noise, an earlier query is a latent and
    # carries none. The query itself is latent too.
    block_noise[:, beam, :width] = np.where(
      survey, fitted.noise[np.clip(picked, 0, structure.size - 1)], 0.0
    )

  kernel = matern52(
    torch.as_tensor(block_points.reshape(-1, width + 1, 2)),
    torch.as_tensor(block_points.reshape(-1, width + 1, 2)),
    log_amplitude,
    log_lengthscale,
  )
  kernel = kernel + torch.diag_embed(
    torch.as_tensor(block_noise.reshape(-1, width + 1)) + floor
  )

  factor = _cholesky(kernel, "query block")

  # The last row of L^-1 is exactly the nonzero part of U's column for this
  # query: the regression weights over its conditioning set, and the reciprocal
  # of its conditional standard deviation.
  basis = torch.zeros(len(factor), width + 1, 1, dtype=torch.float64)
  basis[:, width, 0] = 1.0
  column = torch.linalg.solve_triangular(
    factor.transpose(-1, -2), basis, upper=True
  ).squeeze(-1)
  column = column.reshape(groups, beams, width + 1)

  triangular = torch.zeros(groups, beams, beams, dtype=torch.float64)
  survey_weight = torch.zeros(groups, beams, width, dtype=torch.float64)

  index_tensor = torch.as_tensor(indices)
  survey_mask = torch.as_tensor(from_survey)

  for beam in range(beams):
    triangular[:, beam, beam] = column[:, beam, width]
    survey_weight[:, beam] = torch.where(
      survey_mask[:, beam], column[:, beam, :width], torch.zeros(())
    )
    # Entries at earlier-query rows are V's off-diagonal; scatter them home.
    query_rows = torch.where(
      survey_mask[:, beam],
      torch.zeros_like(index_tensor[:, beam]),
      index_tensor[:, beam],
    )
    contribution = torch.where(
      survey_mask[:, beam], torch.zeros(()), column[:, beam, :width]
    )
    triangular[:, :, beam].scatter_add_(1, query_rows, contribution)

  return triangular, survey_weight, indices, order


def _predict_joint_array(
  fitted: VecchiaMap,
  queries: np.ndarray,
  observation_noise: bool,
  jitter: float,
  beta_uncertainty: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """``predict_joint`` over a ``(G, B, 2)`` stack. Returns ``(G, B)``, ``(G, B, B)``."""
  groups, beams = queries.shape[:2]

  triangular, survey_weight, indices, order = _sparse_columns(
    fitted, queries, jitter
  )

  residual = torch.as_tensor(fitted.residual, dtype=torch.float64)
  survey_basis = torch.as_tensor(
    design_matrix(fitted.structure.points, fitted.basis), dtype=torch.float64
  )
  gathered = torch.as_tensor(indices)

  # ``U_op^T r`` and ``U_op^T H`` -- the same gather-and-dot over each query's
  # survey neighbours, applied to the residual and to the design matrix.
  weighted_residual = (survey_weight * residual[gathered]).sum(-1)
  weighted_basis = (survey_weight[..., None] * survey_basis[gathered]).sum(-2)

  lower = triangular.transpose(-1, -2)
  correction = torch.linalg.solve_triangular(
    lower, weighted_residual[..., None], upper=False
  ).squeeze(-1)

  ordered_queries = torch.as_tensor(
    np.take_along_axis(queries, order[:, :, None], axis=1), dtype=torch.float64
  )
  query_basis = torch.as_tensor(
    design_matrix(ordered_queries.reshape(-1, 2).numpy(), fitted.basis),
    dtype=torch.float64,
  ).reshape(groups, beams, -1)
  beta = torch.as_tensor(fitted.beta, dtype=torch.float64)

  mean = query_basis @ beta - correction

  inverse = torch.linalg.solve_triangular(
    triangular,
    torch.eye(beams, dtype=torch.float64).expand(groups, beams, beams),
    upper=True,
  )
  covariance = inverse.transpose(-1, -2) @ inverse

  if beta_uncertainty and fitted.information is not None:
    # Universal kriging: beta was estimated, so its uncertainty belongs in the
    # prediction. Without this the map understates itself exactly where the
    # queries sit far from the survey's centre of mass.
    residual_basis = query_basis + torch.linalg.solve_triangular(
      lower, weighted_basis, upper=False
    )
    information = torch.as_tensor(fitted.information, dtype=torch.float64)
    covariance = covariance + residual_basis @ torch.linalg.solve(
      information, residual_basis.transpose(-1, -2)
    )

  if observation_noise:
    covariance = covariance + fitted.hyper.noise * torch.eye(
      beams, dtype=torch.float64
    )

  covariance = 0.5 * (covariance + covariance.transpose(-1, -2))

  # Back into the caller's beam order, group by group.
  inverse_order = np.empty_like(order)
  np.put_along_axis(
    inverse_order, order, np.tile(np.arange(beams), (groups, 1)), axis=1
  )

  restored_mean = np.take_along_axis(mean.numpy(), inverse_order, axis=1)
  restored_cov = np.take_along_axis(
    covariance.numpy(), inverse_order[:, :, None], axis=1
  )
  restored_cov = np.take_along_axis(
    restored_cov, inverse_order[:, None, :], axis=2
  )

  return restored_mean, restored_cov
