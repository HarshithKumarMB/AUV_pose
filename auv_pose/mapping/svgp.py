"""Sparse variational GP bathymetry map: the baseline the Vecchia map is scored against."""

from collections.abc import Iterator
from typing import Literal, cast, overload

import gpytorch
import numpy as np
import torch
from gpytorch.utils.memoize import clear_cache_hook
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader, TensorDataset


def resolve_device(device: str | torch.device | None) -> torch.device:
  """The given device, or CUDA when available."""
  if device is not None:
    return torch.device(device)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SVGPModel(gpytorch.models.ApproximateGP):
  """Variational GP, constant mean, RBF kernel with a lengthscale per axis.

  Per axis because inputs are standardised: one shared lengthscale would be
  anisotropic in metres by the survey box's aspect ratio.
  """

  def __init__(self, inducing_points: torch.Tensor) -> None:
    variational_distribution = (
      gpytorch.variational.CholeskyVariationalDistribution(
        inducing_points.size(0)
      )
    )
    variational_strategy = gpytorch.variational.VariationalStrategy(
      self,
      inducing_points,
      variational_distribution,
      learn_inducing_locations=True,
    )
    super().__init__(variational_strategy)

    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
    )

  def forward(
    self, x: torch.Tensor
  ) -> gpytorch.distributions.MultivariateNormal:
    # gpytorch types every Module call as Tensor | Distribution | LinearOperator;
    # a ConstantMean returns a Tensor.
    return gpytorch.distributions.MultivariateNormal(
      cast(torch.Tensor, self.mean_module(x)), self.covar_module(x)
    )


def fit_svgp(
  points: ArrayLike,
  depth: ArrayLike,
  n_inducing: int = 500,
  epochs: int = 200,
  batch_size: int = 5000,
  learning_rate: float = 0.01,
  seed: int | None = None,
  device: str | torch.device | None = None,
) -> "BathymetryMap":
  """Fit the map to soundings by maximising the variational ELBO.

  Inputs and depths are standardised for the fit; the returned map takes and
  gives metres.

  Args:
      points: ``(n, 2)`` sounding positions, metres.
      depth: ``(n,)`` seabed elevations, metres.
      n_inducing: Inducing points, drawn from the soundings.
      epochs: Passes over the data; check ``elbo_trace`` for convergence.
      batch_size: Minibatch size.
      learning_rate: Adam step size.
      seed: Seed for the inducing point draw and the minibatch order.
      device: Where to fit; CUDA when available.
  """
  points = np.asarray(points, dtype=np.float64)
  depth = np.asarray(depth, dtype=np.float64)
  if n_inducing > len(points):
    raise ValueError(
      f"n_inducing={n_inducing} exceeds the {len(points)} training points"
    )
  x_mean, x_scale = points.mean(axis=0), points.std(axis=0)
  y_mean, y_std = float(depth.mean()), float(depth.std())

  device = resolve_device(device)
  train_x = torch.as_tensor((points - x_mean) / x_scale, dtype=torch.float32)
  train_y = torch.as_tensor((depth - y_mean) / y_std, dtype=torch.float32)
  train_x, train_y = train_x.to(device), train_y.to(device)

  # Every random draw -- inducing points, the variational mean's initial jitter,
  # the minibatch order -- comes from torch's generator, seeded for the fit
  # alone so the caller's global state is left as it was.
  with torch.random.fork_rng(enabled=seed is not None):
    if seed is not None:
      torch.manual_seed(seed)
    chosen = torch.randperm(len(train_x))[:n_inducing].to(device)
    model = SVGPModel(train_x[chosen].clone()).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    trace = _train(
      model, likelihood, train_x, train_y, epochs, batch_size, learning_rate
    )

  bathymetry = BathymetryMap(
    model.cpu(), likelihood.cpu(), x_mean, x_scale, y_mean, y_std
  )
  bathymetry.elbo_trace = trace
  bathymetry.fit_device = str(device)
  return bathymetry


def _train(
  model: SVGPModel,
  likelihood: gpytorch.likelihoods.GaussianLikelihood,
  train_x: torch.Tensor,
  train_y: torch.Tensor,
  epochs: int,
  batch_size: int,
  learning_rate: float,
) -> list[float]:
  """Adam on the negative ELBO; the mean loss of each epoch."""
  model.train()
  likelihood.train()
  optimizer = torch.optim.Adam(
    [{"params": model.parameters()}, {"params": likelihood.parameters()}],
    lr=learning_rate,
  )
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
  loader = DataLoader(
    TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
  )
  trace: list[float] = []
  for _ in range(epochs):
    total, batches = 0.0, 0
    for x_batch, y_batch in loader:
      optimizer.zero_grad()
      loss = -cast(torch.Tensor, mll(model(x_batch), y_batch))
      loss.backward()
      optimizer.step()
      total += float(loss.detach())
      batches += 1
    trace.append(total / max(batches, 1))
  return trace


class BathymetryMap:
  """A fitted SVGP taking positions and giving depths in metres.

  :param x_mean: Per-axis mean the inputs were standardised by.
  :param x_scale: Per-axis standard deviation, likewise.
  :param device: Where to evaluate. CPU by default: single-point queries cost
      more in transfer than a GPU saves.
  """

  elbo_trace: list[float]
  fit_device: str = "cpu"

  def __init__(
    self,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_mean: ArrayLike,
    x_scale: ArrayLike,
    y_mean: float,
    y_std: float,
    device: str | torch.device | None = "cpu",
  ) -> None:
    self.device = resolve_device(device)
    self.model = model.to(self.device)
    self.likelihood = likelihood.to(self.device)
    self.x_mean = np.asarray(x_mean, dtype=np.float64)
    self.x_scale = np.asarray(x_scale, dtype=np.float64)
    self.y_mean = float(y_mean)
    self.y_std = float(y_std)
    self.elbo_trace = []

    self.model.eval()
    self.likelihood.eval()

  def _standardise(self, points: NDArray) -> torch.Tensor:
    scaled = (points - self.x_mean) / self.x_scale
    return torch.as_tensor(scaled, dtype=torch.float32, device=self.device)

  def _chunks(self, points: NDArray, chunk_size: int) -> Iterator[torch.Tensor]:
    tensor = self._standardise(points)
    for i in range(0, len(tensor), chunk_size):
      yield tensor[i : i + chunk_size]

  @overload
  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = ...,
    with_std: Literal[False] = ...,
    observation_noise: bool = ...,
  ) -> NDArray: ...

  @overload
  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = ...,
    *,
    with_std: Literal[True],
    observation_noise: bool = ...,
  ) -> tuple[NDArray, NDArray]: ...

  def predict(
    self,
    points: ArrayLike,
    chunk_size: int = 5000,
    with_std: bool = False,
    observation_noise: bool = False,
  ):
    """Predict seabed depth at horizontal positions.

    Args:
        points: ``(n, 2)`` array of ``(x, y)`` in metres.
        chunk_size: Points per forward pass, to bound memory on large grids.
        with_std: Also return the posterior standard deviation, in metres.
        observation_noise: Add the likelihood's noise to that standard
            deviation: the spread of a sounding rather than of the seabed.

    Returns:
        Depths ``(n,)``, or ``(depths, stds)`` when ``with_std``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    if points.shape[1] != 2:
      raise ValueError(f"expected (n, 2) points, got {points.shape}")

    means: list[NDArray] = []
    stds: list[NDArray] = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      for chunk in self._chunks(points, chunk_size):
        latent = self.model(chunk)
        pred = self.likelihood(latent) if observation_noise else latent
        means.append(pred.mean.cpu().numpy())
        if with_std:
          stds.append(pred.stddev.cpu().numpy())

    depth = np.concatenate(means) * self.y_std + self.y_mean
    if not with_std:
      return depth
    # Scaling is affine, so the standard deviation only picks up the scale.
    return depth, np.concatenate(stds) * self.y_std

  def predict_joint(
    self, points: ArrayLike, observation_noise: bool = True
  ) -> tuple[NDArray, NDArray]:
    """Joint Gaussian over the seabed at a set of horizontal positions.

    Implements :class:`~auv_pose.estimation.terrain.DepthMap`, which is what
    the smoother's update step consumes.

    Args:
        points: ``(..., b, 2)`` of ``(x, y)`` in metres. Leading axes are batch
            dimensions -- a ``(31, 32, 2)`` sigma-point cloud comes back as
            ``(31, 32)`` and ``(31, 32, 32)`` from one forward pass.
        observation_noise: Include the likelihood's noise, giving the spread of
            a *sounding* rather than of the seabed.

    Returns:
        ``(mean, cov)`` of shapes ``(..., b)`` and ``(..., b, b)``, in metres
        and metres squared.

    Note:
        No ``fast_pred_var``: its low-rank covariance gets the off-diagonals
        wrong, and they are what this method is for.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1] != 2:
      raise ValueError(f"expected (..., b, 2) points, got {points.shape}")

    tensor = self._standardise(points)

    with torch.no_grad():
      latent = self.model(tensor)
      predicted = cast(
        gpytorch.distributions.MultivariateNormal,
        self.likelihood(latent) if observation_noise else latent,
      )
      mean = predicted.mean.cpu().numpy()
      cov = predicted.covariance_matrix.cpu().numpy()

    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    return mean * self.y_std + self.y_mean, cov * self.y_std**2

  def mean_gradient(self, points: ArrayLike) -> NDArray:
    """Slope of the map's mean, in metres per metre.

    Args:
        points: ``(n, 2)`` of ``(x, y)`` in metres.

    Returns:
        ``(n, 2)`` of ``d(depth)/dx, d(depth)/dy``.

    Note:
        The cache is cleared first: ``VariationalStrategy`` memoises, and a
        second backward pass through the memo raises.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    if points.shape[-1] != 2:
      raise ValueError(f"expected (n, 2) points, got {points.shape}")

    tensor = self._standardise(points).requires_grad_(True)

    clear_cache_hook(self.model.variational_strategy)
    (gradient,) = torch.autograd.grad(self.model(tensor).mean.sum(), tensor)

    # Out of the standardised input and target, back into metres per metre.
    return gradient.cpu().numpy() * self.y_std / self.x_scale
