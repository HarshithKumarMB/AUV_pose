"""Sparse variational GP surrogate for seabed bathymetry.

Maps horizontal position ``(x, y)`` to seabed depth. An exact GP is not an option here: the surveys hold ~82k soundings and exact
inference is cubic in that. A sparse variational GP with a few hundred inducing
points is.
"""

from __future__ import annotations

from collections.abc import Iterator

import gpytorch
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["BathymetryMap", "SVGPModel", "fit_svgp", "resolve_device"]


def resolve_device(device: str | torch.device | None) -> torch.device:
  """Pick a torch device, defaulting to CUDA when it is genuinely available.

  ``torch.cuda.is_available()`` is False on a CPU-only build however capable
  the card is, so this silently falls back rather than failing -- but a fit that
  was expected to be on the GPU and quietly was not is worth noticing, which is
  why :func:`fit_svgp` reports the device it chose.
  """
  if device is not None:
    return torch.device(device)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SVGPModel(gpytorch.models.ApproximateGP):
  """Stochastic variational GP with a constant mean and a scaled RBF kernel.

  Inducing point locations are learned along with the variational parameters.

  **The kernel is ARD**, one lengthscale per input axis. A single isotropic
  lengthscale is not the neutral choice it looks like: inputs are standardised
  before fitting, so an isotropic kernel is implicitly anisotropic in metres by
  exactly the ratio of the two scalers -- which is a fact about the shape of the
  surveyed box, not about the seabed. The map fitted that way reported
  lengthscales of 2.13 m by 1.19 m purely because the survey area is 40 m by
  20 m.

  :ivar elbo_trace: Mean negative ELBO per epoch, set by :func:`fit_svgp`.
  :ivar fit_device: Device the fit ran on, set by :func:`fit_svgp`.
  """

  elbo_trace: list[float]
  fit_device: str

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
    return gpytorch.distributions.MultivariateNormal(
      self.mean_module(x), self.covar_module(x)
    )


def fit_svgp(
  train_x: torch.Tensor,
  train_y: torch.Tensor,
  n_inducing: int = 500,
  epochs: int = 200,
  batch_size: int = 5000,
  learning_rate: float = 0.01,
  seed: int | None = None,
  device: str | torch.device | None = None,
) -> tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor]:
  """Fit an SVGP by maximising the variational ELBO.

  Args:
      train_x: Standardised inputs, ``(n, 2)``.
      train_y: Standardised targets, ``(n,)``.
      n_inducing: Number of inducing points, sampled from the training inputs.
      epochs: Passes over the data. The default was 20, which on the survey
          soundings is about 340 Adam steps and nowhere near converged: going to
          200 moved held-out rmse from 1.25 m to 1.08 m, the fitted noise from
          1.30 to 1.10, and the lengthscales by nearly a factor of two. Check
          :attr:`SVGPModel.elbo_trace` rather than assuming any number here is
          enough.
      batch_size: Minibatch size.
      learning_rate: Adam step size.
      seed: Seed for inducing point selection, for reproducible fits.
      device: Where to fit. Defaults to CUDA when available -- the cost is
          dominated by Cholesky factorisations of the inducing covariance, which
          the GPU is much better at, and the returned model is moved back to the
          CPU so checkpoints stay portable.

  Returns:
      ``(model, likelihood, inducing_points)``, all in training mode and on the
      CPU. The model carries ``elbo_trace``, the mean negative ELBO per epoch,
      so convergence is observable instead of assumed, and ``fit_device``,
      recording where it was actually fitted.
  """
  if n_inducing > len(train_x):
    raise ValueError(
      f"n_inducing={n_inducing} exceeds the {len(train_x)} training points"
    )

  device = resolve_device(device)
  train_x = train_x.to(device)
  train_y = train_y.to(device)

  rng = np.random.default_rng(seed)
  idx = rng.choice(len(train_x), n_inducing, replace=False)
  inducing_points = train_x[idx].clone()

  model = SVGPModel(inducing_points).to(device)
  likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

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
    epoch_loss = 0.0
    batches = 0
    for x_batch, y_batch in loader:
      optimizer.zero_grad()
      loss = -mll(model(x_batch), y_batch)
      loss.backward()
      optimizer.step()
      epoch_loss += float(loss.detach())
      batches += 1
    trace.append(epoch_loss / max(batches, 1))

  model.elbo_trace = trace
  model.fit_device = str(device)

  # Back to the CPU: a checkpoint holding CUDA tensors only loads on a machine
  # with a GPU, and prediction is single points at tick rate where the transfer
  # would cost more than the arithmetic saves.
  return model.cpu(), likelihood.cpu(), inducing_points.cpu()


class BathymetryMap:
  """A fitted SVGP plus the scaling needed to use it in metric units.

  The model is trained on standardised inputs and targets; this wraps it so
  callers can pass raw ``(x, y)`` and get depths back in metres.

  :param device: Where to evaluate. Defaults to the CPU deliberately, unlike
      :func:`fit_svgp`: ``navigate.py`` queries this one point at a time at tick
      rate, where moving that point to a GPU and the answer back costs far more
      than the arithmetic saves. Pass ``"cuda"`` when scoring a large held-out
      set or rendering a grid, which is where it does pay.

  Note:
      ``torch.nn.Module.to`` moves in place, so passing a non-CPU device moves
      the caller's ``model`` too rather than taking a copy. That is ordinary
      PyTorch behaviour, but it means a checkpoint written afterwards would hold
      CUDA tensors -- :func:`auv_pose.io.checkpoints.save_map` forces them back
      to the CPU for exactly this reason.
  """

  def __init__(
    self,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_scaler,
    y_mean: float,
    y_std: float,
    device: str | torch.device | None = "cpu",
  ) -> None:
    self.device = resolve_device(device)
    self.model = model.to(self.device)
    self.likelihood = likelihood.to(self.device)
    self.x_scaler = x_scaler
    self.y_mean = float(y_mean)
    self.y_std = float(y_std)

    self.model.eval()
    self.likelihood.eval()

  def _chunks(self, points: NDArray, chunk_size: int) -> Iterator[torch.Tensor]:
    scaled = self.x_scaler.transform(points)
    tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
    for i in range(0, len(tensor), chunk_size):
      yield tensor[i : i + chunk_size]

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
            deviation, giving the spread of a *sounding* rather than of the
            seabed. Off by default: a caller asking a map how deep the seabed
            is wants to know how well the seabed is known there.

    Returns:
        Depths ``(n,)``, or ``(depths, stds)`` when ``with_std``.

    Note:
        The distinction matters more than it looks. This previously always
        added the noise, and since a poorly fitted GP absorbs its own misfit
        into that noise term, the number it returned was ~95% noise and nearly
        constant across the map -- 1.36 to 1.87 m against a fitted noise of
        1.36 m. As a measure of "where is this map trustworthy" that is
        useless, and it is exactly what a filter wanting to weight the map
        against its other sensors would have consumed.
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
