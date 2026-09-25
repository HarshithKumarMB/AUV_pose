"""The Vecchia approximation: a Gaussian process that scales to a whole survey.

Each sounding conditions on its ``m`` nearest predecessors under a maximin
ordering::

    p(z) = prod_i p(z_i | z_{g(i)}),    g(i) subset of {1..i-1},  |g(i)| <= m

The map is ``z = phi(q)^T beta + zeta(q) + eps``; this module models the
residual ``zeta`` of the mean, so targets and inputs are never standardised.

Shared parameters: ``log_amplitude`` is ``log(sigma_f^2)``, ``log_lengthscale``
the ``log`` of the per-axis lengthscale in metres, ``noise`` the per-sounding
variance (a scalar broadcasts), ``jitter`` a diagonal regulariser relative to
the amplitude, and ``chunk``/``chunk_size`` a batch size that bounds memory
only. Arrays are in the structure's order unless stated.

Everything is float64: soundings 0.25 m apart under a metre-scale kernel give
kernel blocks too ill-conditioned for float32.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Self, overload

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve_triangular
from scipy.spatial import KDTree
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from auv_pose.mapping.kernels import matern52, matern52_gradient
from auv_pose.mapping.ordering import maximin_order, ordered_neighbours


def _resolve_device(device: str | torch.device | None) -> torch.device:
  """CUDA when available, else the CPU. Local copy to avoid importing gpytorch."""
  if device is not None:
    return torch.device(device)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_LOG_2PI = math.log(2.0 * math.pi)

#: Diagonal regulariser for every kernel block, relative to the amplitude. Not
#: a physical nugget: it keeps near-coincident soundings factorisable.
DEFAULT_JITTER = 1e-8


@dataclass(frozen=True)
class VecchiaStructure:
  """Ordering and conditioning sets: the position-only part, reused across a fit.

  :param points: Sounding positions **in the approximation's own order**,
      shape ``(N, 2)``, metres.
  :param neighbours: Indices into :attr:`points` of each sounding's
      conditioning set, shape ``(N, m)``, padded with ``-1``. Rows below
      :attr:`n0` are unused.
  :param order: The permutation taking the caller's ordering to this one.
  :param n0: Size of the dense head block. At least ``m``, so every batched
      row has exactly ``m`` neighbours.
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
  :param near: How many of the ``m`` are nearest neighbours; see
      :func:`~auv_pose.mapping.ordering.ordered_neighbours`. Defaults to all
      nearest.
  :return: The structure, with ``points`` permuted into maximin order.
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
  """The mean function's basis: a polynomial in position, the plane by default.

  :param degree: Total degree; 1 is the plane.
  """

  kind: str = "polynomial"
  degree: int = 1

  @classmethod
  def build(cls, kind: str = "linear", degree: int | None = None) -> Self:
    """Choose a basis.

    :param kind: ``"linear"``, ``"quadratic"``, ``"cubic"`` or ``"polynomial"``.
    :param degree: Overrides the degree implied by ``kind``.
    """
    named = {"linear": 1, "quadratic": 2, "cubic": 3}
    if kind in named:
      return cls(kind="polynomial", degree=degree or named[kind])
    if kind == "polynomial":
      return cls(kind="polynomial", degree=degree or 1)
    raise ValueError(
      f"kind must be one of {sorted(named) + ['polynomial']}, got {kind!r}"
    )

  @property
  def size(self) -> int:
    """Number of basis functions, ``p``."""
    return (self.degree + 1) * (self.degree + 2) // 2

  def __call__(self, points: ArrayLike) -> np.ndarray:
    """Evaluate the basis at ``(N, 2)`` positions. Shape ``(N, p)``."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
      raise ValueError(f"expected (N, 2) points, got {points.shape}")

    columns = [np.ones(len(points))]
    for total in range(1, self.degree + 1):
      for power in range(total + 1):
        columns.append(points[:, 0] ** (total - power) * points[:, 1] ** power)
    return np.column_stack(columns)

  def gradient(self, points: ArrayLike) -> np.ndarray:
    """Derivative of the basis with respect to position. Shape ``(N, p, 2)``."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
      raise ValueError(f"expected (N, 2) points, got {points.shape}")

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


#: The default mean, ``phi(q) = (1, n, e)``.
LINEAR_MEAN = MeanBasis(kind="polynomial", degree=1)


def design_matrix(
  points: ArrayLike, basis: MeanBasis | None = None
) -> np.ndarray:
  """Evaluate a mean basis, :data:`LINEAR_MEAN` by default. Shape ``(N, p)``."""
  return (basis or LINEAR_MEAN)(points)


def _gather_block(values: Tensor, neighbours: Tensor, rows: Tensor) -> Tensor:
  """Gather a per-sounding quantity over ``(g(i), i)`` -- neighbours, self last.

  Self last means the last row of ``chol(K)`` holds the regression weights and
  the conditional standard deviation, so one Cholesky gives both.

  :return: Shape ``(b, m+1)`` or ``(b, m+1, d)``.
  """
  return torch.cat([values[neighbours], values[rows][:, None]], dim=1)


def _cholesky(blocks: Tensor, where: str) -> Tensor:
  """Batched Cholesky that raises, naming which block failed, instead of NaNs."""
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

  A separate function so it can be gradient-checkpointed.
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
  """``(U^T V)^T (U^T V)`` and ``sum_i log U_ii``, in memory bounded by ``chunk``.

  :param values: Shape ``(N, p)``, in the structure's order.
  :return: ``(gram, sum_i log U_ii)`` of shapes ``(p, p)`` and ``()``.

  Note:
      Chunking alone does not bound memory: autograd would keep every chunk's
      factor for the backward pass. Each chunk is gradient-checkpointed instead.
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

      recomputed = checkpoint(
        _gram_chunk,
        _gather_block(points, conditioning, rows),
        _gather_block(values, conditioning, rows),
        _gather_block(noise, conditioning, rows),
        log_amplitude,
        log_lengthscale,
        floor,
        use_reentrant=False,
      )
      # checkpoint returns what the function does; its stub only says Any|None.
      assert recomputed is not None
      piece, diagonal = recomputed
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

  ``U`` is the sparse upper-triangular factor with ``K^-1 = U U^T``; it is
  never assembled.

  :param values: Shape ``(N,)`` or ``(N, p)``, in the structure's order.
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

  :return: ``U`` as a ``scipy.sparse`` CSC matrix, shape ``(N, N)``, upper
      triangular, in the **structure's** order.

  Note:
      With regression weights ``w`` and conditional standard deviation ``d``,
      ``U[i, i] = 1/d`` and ``U[g(i), i] = -w/d``. The head block's corner is
      ``L^-T``.
  """
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

  :return: Shape ``(N,)`` when ``count`` is 1, else ``(N, count)``, in the
      **caller's** order, matching the positions given to
      :func:`build_structure`.

  Note:
      To test hyperparameter recovery, draw with a conditioning set several
      times larger than the one being fitted; the same ``m`` is circular.
  """
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

  ``log p = -N/2 log 2pi + sum_i log U_ii - 1/2 ||U^T r||^2``.

  With a mean estimated from the same data this underestimates the amplitude;
  prefer :func:`vecchia_reml`.

  :param residual: ``z - phi(q)^T beta`` in the structure's order, shape ``(N,)``.
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

  :param depth: Seabed elevation in the structure's order, shape ``(N,)``.
  :param basis: The linear mean's design matrix ``phi(q)``, shape ``(N, p)``,
      in the same order.
  :return: ``(restricted log likelihood, beta)``, with ``beta`` the GLS
      estimate under the current kernel.

  Note:
      ``beta`` must be recomputed at every hyperparameter value; holding it
      fixed turns this back into plug-in maximum likelihood.

  Note:
      The constant ``+1/2 log|H'H|`` is omitted, so values are comparable
      across kernels but **not** across designs ``H``.
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
  """Kernel parameters, held in logs as they are fitted.

  :param log_noise: ``log(sigma_z^2)``, the base per-sounding variance.
  """

  log_amplitude: float
  log_lengthscale: tuple[float, float]
  log_noise: float

  def kernel_tensors(
    self, device: torch.device | str | None = None
  ) -> tuple[Tensor, Tensor]:
    """``(log_amplitude, log_lengthscale)`` as :func:`matern52` takes them."""
    return (
      torch.tensor(self.log_amplitude, dtype=torch.float64, device=device),
      torch.tensor(self.log_lengthscale, dtype=torch.float64, device=device),
    )

  @property
  def amplitude(self) -> float:
    """``sigma_f^2``, metres squared."""
    return math.exp(self.log_amplitude)

  @property
  def lengthscale(self) -> np.ndarray:
    """Per-axis correlation length, metres."""
    return np.exp(np.asarray(self.log_lengthscale, dtype=float))

  @property
  def noise(self) -> float:
    """``sigma_z^2``, metres squared."""
    return math.exp(self.log_noise)


@dataclass
class VecchiaMap:
  """A fitted seabed map.

  :param residual: ``z - phi(q)^T beta`` in the structure's order, shape ``(N,)``.
  :param beta: Mean coefficients, ``(intercept, d/dn, d/de)`` for the plane.
  :param noise: Per-sounding variance, shape ``(N,)``, in the structure's
      order.
  :param loglik_trace: Objective at each optimiser step.
  :param information: ``H' K^-1 H``, shape ``(p, p)``; prediction needs it to
      propagate the uncertainty in ``beta``.
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
  _tree: KDTree | None = field(default=None, repr=False, compare=False)

  @property
  def lengthscale(self) -> np.ndarray:
    """Per-axis correlation length in metres. See :class:`VecchiaHyperparameters`."""
    return self.hyper.lengthscale

  @property
  def tree(self) -> KDTree:
    """A ``KDTree`` over the survey, built on first use and kept."""
    tree = self._tree
    if tree is None:
      tree = self._tree = KDTree(self.structure.points)
    return tree

  def predict_joint(
    self,
    points: ArrayLike,
    observation_noise: bool = True,
    jitter: float = DEFAULT_JITTER,
    beta_uncertainty: bool = True,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Joint Gaussian over the seabed at a set of horizontal positions.

    Implements :class:`~auv_pose.estimation.terrain.DepthMap`. Queries condition
    on each other as well as on the survey, so the covariance is full.

    :param points: ``(..., b, 2)`` of ``(x, y)`` in metres; leading axes are
        batch dimensions. ``b`` must not exceed ``m``.
    :param observation_noise: Include ``sigma_z^2``, giving the spread of a
        *sounding* rather than of the seabed.
    :param beta_uncertainty: Propagate the uncertainty in the mean. Leave on.
    :return: ``(mean, cov)`` of shapes ``(..., b)`` and ``(..., b, b)``, metres
        and metres squared. Depth is **z-up**: a seabed 65 m down is ``-65``.
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
    """Slope of the map's mean, in metres per metre, one point at a time.

    Implements :class:`~auv_pose.estimation.terrain.DepthMap`. Analytic::

        grad mu(q) = [d phi(q)/dq]' beta + [dk(q, g)/dq]' K_gg^-1 r_g

    :param points: ``(n, 2)`` of ``(x, y)`` in metres.
    :return: ``(n, 2)`` of ``d(elevation)/dx, d(elevation)/dy``.

    Note:
        The mean is only piecewise smooth: the conditioning set jumps between
        soundings, so compare against finite differences only at full
        conditioning.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[-1] != 2:
      raise ValueError(f"expected (n, 2) points, got {points.shape}")

    structure = self.structure
    width = min(structure.conditioning, structure.size)

    log_amplitude, log_lengthscale = self.hyper.kernel_tensors()
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

  @overload
  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = ...,
    with_std: Literal[False] = ...,
    observation_noise: bool = ...,
    jitter: float = ...,
  ) -> np.ndarray: ...

  @overload
  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = ...,
    *,
    with_std: Literal[True],
    observation_noise: bool = ...,
    jitter: float = ...,
  ) -> tuple[np.ndarray, np.ndarray]: ...

  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = 5000,
    with_std: bool = False,
    observation_noise: bool = False,
    jitter: float = DEFAULT_JITTER,
  ):
    """Predict seabed elevation at horizontal positions, one point at a time.

    :param points: ``(n, 2)`` of ``(x, y)`` in metres.
    :param with_std: Also return the posterior standard deviation, in metres.
    :param observation_noise: Add ``sigma_z^2``, giving the spread of a
        sounding rather than of the seabed.
    :return: Elevations ``(n,)``, or ``(elevation, std)`` when ``with_std``.

    Note:
        Each query conditions on the survey alone, so this differs slightly
        from the diagonal of :meth:`predict_joint`.
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
  """A data-driven starting point for the fit.

  :param points: Sounding positions, shape ``(N, 2)``.
  :param residual: Survey depths with the mean removed, shape ``(N,)``.
  :param ard: Per-axis lengthscales; when ``False`` both axes start equal.
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
  steps: int = 300,
  learning_rate: float = 0.05,
  ard: bool = True,
  method: str = "reml",
  device: str | torch.device | None = None,
  jitter: float = DEFAULT_JITTER,
  chunk: int = 8192,
  near: int | None = None,
  mean: str = "linear",
) -> VecchiaMap:
  """Fit the mean and the kernel hyperparameters to a survey.

  ``method="reml"`` profiles ``beta`` out by GLS at every step;
  ``method="ml"`` fixes ``beta`` by least squares first.

  :param points: Sounding positions, shape ``(N, 2)``, metres, any order.
  :param depth: Seabed elevation at each, shape ``(N,)``. **z-up**, so a seabed
      65 m down is ``-65``.
  :param n0: Dense head-block size; see :func:`build_structure`.
  :param learning_rate: Adam step size, on the log parameters.
  :param method: ``"reml"`` or ``"ml"``; see :func:`vecchia_reml`.
  :param near: How many of the ``m`` are nearest neighbours, the rest spread
      across the ordering; ``None`` is all nearest.
  :param mean: Mean basis; see :meth:`MeanBasis.build`.
  :return: The fitted map. Deterministic for a given survey and settings.
  """
  points = np.asarray(points, dtype=float)
  depth = np.asarray(depth, dtype=float)
  if len(points) != len(depth):
    raise ValueError(f"{len(points)} positions against {len(depth)} depths")
  if method not in ("reml", "ml"):
    raise ValueError(f'method must be "reml" or "ml", got {method!r}')

  structure = build_structure(points, m=m, n0=n0, near=near)
  ordered_points = structure.points
  ordered_depth = depth[structure.order]

  mean_basis = MeanBasis.build(kind=mean)
  basis = design_matrix(ordered_points, mean_basis)
  # Only ever a starting point under REML, where the objective refits it.
  beta = np.linalg.lstsq(basis, ordered_depth, rcond=None)[0]
  residual = ordered_depth - basis @ beta

  start = initial_hyperparameters(ordered_points, residual, ard=ard)
  resolved = _resolve_device(device)

  def tensor(values) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float64, device=resolved)

  depth_tensor, basis_tensor = tensor(ordered_depth), tensor(basis)
  residual_tensor = tensor(residual)

  start_amplitude, start_lengthscale = start.kernel_tensors(resolved)
  log_amplitude = start_amplitude.clone().requires_grad_(True)
  log_noise = tensor(start.log_noise).clone().requires_grad_(True)
  # With ard off there is one lengthscale, broadcast to both axes, so the
  # optimiser cannot pull them apart.
  log_lengthscale = (
    (start_lengthscale if ard else start_lengthscale[:1])
    .clone()
    .requires_grad_(True)
  )

  def per_axis() -> Tensor:
    return log_lengthscale if ard else log_lengthscale.expand(2)

  def objective() -> tuple[Tensor, Tensor | None]:
    variance = torch.exp(log_noise)
    if method == "reml":
      return vecchia_reml(
        structure,
        depth_tensor,
        basis_tensor,
        log_amplitude,
        per_axis(),
        variance,
        jitter=jitter,
        chunk=chunk,
      )
    value = vecchia_loglik(
      structure,
      residual_tensor,
      log_amplitude,
      per_axis(),
      variance,
      jitter=jitter,
      chunk=chunk,
    )
    return value, None

  optimiser = torch.optim.Adam(
    [log_amplitude, log_lengthscale, log_noise], lr=learning_rate
  )
  trace: list[float] = []
  for _ in range(steps):
    optimiser.zero_grad()
    value, _ = objective()
    (-value).backward()
    optimiser.step()
    trace.append(float(value.detach()))

  with torch.no_grad():
    if method == "reml":
      # One last evaluation, so beta matches the hyperparameters returned
      # rather than the ones from the step before the final update.
      _, fitted_beta = objective()
      assert fitted_beta is not None
      beta = fitted_beta.cpu().numpy()
      residual = ordered_depth - basis @ beta

    final = per_axis()
    gram, _ = whiten_gram(
      structure,
      basis_tensor,
      log_amplitude,
      final,
      torch.exp(log_noise),
      jitter=jitter,
      chunk=chunk,
    )
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
    noise=np.full(len(points), fitted.noise),
    loglik_trace=trace,
    basis=mean_basis,
    fit_device=str(resolved),
    information=gram.cpu().numpy(),
  )


def _query_order(queries: np.ndarray) -> np.ndarray:
  """Maximin order within each group of queries, shape ``(G, B, 2)`` -> ``(G, B)``.

  Ordered per group, not once per batch, so results do not depend on how the
  caller batched them. Maximin keeps a fan's beams from conditioning only on
  one side.
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
) -> tuple[Tensor, Tensor, np.ndarray, np.ndarray]:
  """The nonzero entries of ``U``'s query columns, one block at a time.

  With survey first and queries last, the conditional's factor ``V = U_pp`` is
  a submatrix of ``U``; do not form and refactorise ``U U^T`` instead.

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

  log_amplitude, log_lengthscale = fitted.hyper.kernel_tensors()
  floor = jitter * fitted.hyper.amplitude

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

    # Survey neighbours carry sounding noise; earlier queries and the query
    # itself are latent and carry none.
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
    # Universal kriging: add the uncertainty of the estimated beta.
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
