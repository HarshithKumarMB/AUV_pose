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

__all__ = ["BathymetryMap", "SVGPModel", "fit_svgp"]


class SVGPModel(gpytorch.models.ApproximateGP):
  """Stochastic variational GP with a constant mean and a scaled RBF kernel.

  Inducing point locations are learned along with the variational parameters.
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
      gpytorch.kernels.RBFKernel()
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
  epochs: int = 20,
  batch_size: int = 5000,
  learning_rate: float = 0.01,
  seed: int | None = None,
) -> tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor]:
  """Fit an SVGP by maximising the variational ELBO.

  Args:
      train_x: Standardised inputs, ``(n, 2)``.
      train_y: Standardised targets, ``(n,)``.
      n_inducing: Number of inducing points, sampled from the training inputs.
      epochs: Passes over the data.
      batch_size: Minibatch size.
      learning_rate: Adam step size.
      seed: Seed for inducing point selection, for reproducible fits.

  Returns:
      ``(model, likelihood, inducing_points)``, all in training mode.
  """
  if n_inducing > len(train_x):
    raise ValueError(
      f"n_inducing={n_inducing} exceeds the {len(train_x)} training points"
    )

  rng = np.random.default_rng(seed)
  idx = rng.choice(len(train_x), n_inducing, replace=False)
  inducing_points = train_x[idx].clone()

  model = SVGPModel(inducing_points)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()

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

  for _ in range(epochs):
    for x_batch, y_batch in loader:
      optimizer.zero_grad()
      loss = -mll(model(x_batch), y_batch)
      loss.backward()
      optimizer.step()

  return model, likelihood, inducing_points


class BathymetryMap:
  """A fitted SVGP plus the scaling needed to use it in metric units.

  The model is trained on standardised inputs and targets; this wraps it so
  callers can pass raw ``(x, y)`` and get depths back in metres.
  """

  def __init__(
    self,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_scaler,
    y_mean: float,
    y_std: float,
  ) -> None:
    self.model = model
    self.likelihood = likelihood
    self.x_scaler = x_scaler
    self.y_mean = float(y_mean)
    self.y_std = float(y_std)

    self.model.eval()
    self.likelihood.eval()

  def _chunks(self, points: NDArray, chunk_size: int) -> Iterator[torch.Tensor]:
    scaled = self.x_scaler.transform(points)
    tensor = torch.as_tensor(scaled, dtype=torch.float32)
    for i in range(0, len(tensor), chunk_size):
      yield tensor[i : i + chunk_size]

  def predict(
    self, points: ArrayLike, chunk_size: int = 5000, with_std: bool = False
  ):
    """Predict seabed depth at horizontal positions.

    Args:
        points: ``(n, 2)`` array of ``(x, y)`` in metres.
        chunk_size: Points per forward pass, to bound memory on large grids.
        with_std: Also return the posterior standard deviation, in metres.

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
        pred = self.likelihood(self.model(chunk))
        means.append(pred.mean.cpu().numpy())
        if with_std:
          stds.append(pred.stddev.cpu().numpy())

    depth = np.concatenate(means) * self.y_std + self.y_mean
    if not with_std:
      return depth
    # Scaling is affine, so the standard deviation only picks up the scale.
    return depth, np.concatenate(stds) * self.y_std
