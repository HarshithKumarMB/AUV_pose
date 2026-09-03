"""Persisting a fitted bathymetry map.

The checkpoint bundles the model weights with the scalers, because a GP fitted on
standardised inputs is unusable without them.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import gpytorch
import torch

from auv_pose.mapping.svgp import BathymetryMap, SVGPModel

__all__ = ["load_map", "save_map"]

_REQUIRED_KEYS = frozenset(
  {
    "model_state_dict",
    "likelihood_state_dict",
    "inducing_points",
    "x_scaler",
    "y_mean",
    "y_std",
  }
)


def save_map(
  path: str | Path,
  model: SVGPModel,
  likelihood: gpytorch.likelihoods.GaussianLikelihood,
  inducing_points: torch.Tensor,
  x_scaler: Any,
  y_mean: float,
  y_std: float,
) -> None:
  """Write a fitted map to ``path`` as a pickle.

  Tensors are forced onto the CPU. ``torch.nn.Module.to`` moves a model in
  place, so anything that has evaluated the map on a GPU -- including
  :class:`~auv_pose.mapping.svgp.BathymetryMap` constructed with
  ``device="cuda"`` -- leaves the caller holding a model whose state dict is
  full of CUDA tensors, and a checkpoint written from that only loads on a
  machine with a GPU.
  """
  payload = {
    "model_state_dict": {
      key: value.cpu() for key, value in model.state_dict().items()
    },
    "likelihood_state_dict": {
      key: value.cpu() for key, value in likelihood.state_dict().items()
    },
    "inducing_points": inducing_points.cpu(),
    "x_scaler": x_scaler,
    "y_mean": float(y_mean),
    "y_std": float(y_std),
  }
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)


def load_map(path: str | Path) -> BathymetryMap:
  """Load a fitted map, ready for prediction.

  Note:
      The checkpoint contains a pickled scikit-learn ``StandardScaler``. Pickles
      are not portable across scikit-learn versions -- if this warns about a
      version mismatch, refit with ``experiments/train_map.py`` rather than
      trusting the loaded scaler, since every depth query passes through it.
  """
  path = Path(path)
  with open(path, "rb") as handle:
    checkpoint = pickle.load(handle)

  missing = _REQUIRED_KEYS - set(checkpoint)
  if missing:
    raise ValueError(
      f"{path} is not a bathymetry checkpoint; missing {sorted(missing)}"
    )

  model = SVGPModel(checkpoint["inducing_points"])
  likelihood = gpytorch.likelihoods.GaussianLikelihood()

  # gpytorch migrates pre-rename ConstantMean checkpoints itself, and warns.
  try:
    model.load_state_dict(checkpoint["model_state_dict"])
  except RuntimeError as error:
    raise ValueError(
      f"{path} does not match the current model. Checkpoints written before "
      "the kernel gained per-axis lengthscales (ARD) store one lengthscale "
      "where two are now expected. Refit with experiments/train_map.py."
    ) from error
  likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

  return BathymetryMap(
    model=model,
    likelihood=likelihood,
    x_scaler=checkpoint["x_scaler"],
    y_mean=checkpoint["y_mean"],
    y_std=checkpoint["y_std"],
  )
