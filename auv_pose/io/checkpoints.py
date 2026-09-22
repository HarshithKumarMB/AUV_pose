"""Persisting a fitted bathymetry map.

Two formats live here. The SVGP checkpoint bundles model weights with the
scalers, because a GP fitted on standardised inputs is unusable without them.
The Vecchia checkpoint carries the survey itself, its ordering and conditioning
sets, and the fitted parameters -- there are no weights and no scalers, because
that map fits in metres.

:func:`load_map` dispatches on a ``format`` field and reads both.

.. note::

   **Older checkpoints have no ``format`` field at all**, so the dispatch falls
   back to recognising an SVGP by its ``model_state_dict``. That fallback is
   permanent: checkpoints already exist without a version marker, which is
   precisely the problem the field is there to stop recurring. The cost of the
   omission is on the record -- :func:`load_map` still carries a bespoke
   handler translating a state-dict shape mismatch into an explanation about
   per-axis lengthscales, because that diagnosis had to be reverse-engineered
   once already.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch

from auv_pose.mapping.svgp import BathymetryMap, SVGPModel
from auv_pose.mapping.vecchia import (
  VecchiaHyperparameters,
  VecchiaMap,
  VecchiaStructure,
)

__all__ = ["load_map", "save_map", "save_vecchia_map"]

#: Bumped when a stored field changes meaning, so a stale checkpoint is
#: refused with an explanation rather than mis-read.
VECCHIA_VERSION = 1

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

_VECCHIA_KEYS = frozenset(
  {
    "points",
    "neighbours",
    "order",
    "n0",
    "residual",
    "beta",
    "noise",
    "log_amplitude",
    "log_lengthscale",
    "log_noise",
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


def save_vecchia_map(path: str | Path, bathymetry: VecchiaMap) -> None:
  """Write a fitted Vecchia map to ``path`` as a pickle.

  Stores the survey, its ordering, its conditioning sets and the fitted
  parameters. Nothing here is a torch tensor or a scikit-learn object, so the
  checkpoint has no framework version to be portable across -- which the SVGP
  format cannot say.

  :param path: Destination.
  :param bathymetry: The fitted map.

  Note:
      ``neighbours`` is stored rather than rebuilt on load. It is the larger
      part of the file -- 17 MB at 144k soundings against 2.3 MB for the
      positions -- but rebuilding it means re-running the ordering, and an
      ordering that came out differently from the stored one would be a
      thoroughly confusing thing to debug. Determinism is cheaper to store
      than to rely on.
  """
  structure = bathymetry.structure
  payload = {
    "format": "vecchia",
    "version": VECCHIA_VERSION,
    "points": np.asarray(structure.points, dtype=np.float64),
    "neighbours": np.asarray(structure.neighbours, dtype=np.int32),
    "order": np.asarray(structure.order, dtype=np.int32),
    "n0": int(structure.n0),
    "residual": np.asarray(bathymetry.residual, dtype=np.float64),
    "beta": np.asarray(bathymetry.beta, dtype=np.float64),
    "noise": np.asarray(bathymetry.noise, dtype=np.float64),
    "log_amplitude": float(bathymetry.hyper.log_amplitude),
    "log_lengthscale": [float(v) for v in bathymetry.hyper.log_lengthscale],
    "log_noise": float(bathymetry.hyper.log_noise),
    "information": (
      None
      if bathymetry.information is None
      else np.asarray(bathymetry.information, dtype=np.float64)
    ),
    "loglik_trace": [float(v) for v in bathymetry.loglik_trace],
    "fit_device": str(bathymetry.fit_device),
  }
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)


def _load_vecchia(path: Path, checkpoint: dict) -> VecchiaMap:
  """Rebuild a :class:`~auv_pose.mapping.vecchia.VecchiaMap` from a payload."""
  version = checkpoint.get("version")
  if version != VECCHIA_VERSION:
    raise ValueError(
      f"{path} is a version {version} Vecchia checkpoint; this build reads "
      f"version {VECCHIA_VERSION}. Refit with experiments/train_map.py."
    )

  missing = _VECCHIA_KEYS - set(checkpoint)
  if missing:
    raise ValueError(
      f"{path} is not a bathymetry checkpoint; missing {sorted(missing)}"
    )

  structure = VecchiaStructure(
    points=np.asarray(checkpoint["points"], dtype=np.float64),
    neighbours=np.asarray(checkpoint["neighbours"], dtype=np.int64),
    order=np.asarray(checkpoint["order"], dtype=np.int64),
    n0=int(checkpoint["n0"]),
  )

  return VecchiaMap(
    structure=structure,
    residual=np.asarray(checkpoint["residual"], dtype=np.float64),
    beta=np.asarray(checkpoint["beta"], dtype=np.float64),
    hyper=VecchiaHyperparameters(
      log_amplitude=float(checkpoint["log_amplitude"]),
      log_lengthscale=tuple(float(v) for v in checkpoint["log_lengthscale"]),
      log_noise=float(checkpoint["log_noise"]),
    ),
    noise=np.asarray(checkpoint["noise"], dtype=np.float64),
    loglik_trace=list(checkpoint.get("loglik_trace", [])),
    fit_device=str(checkpoint.get("fit_device", "cpu")),
    information=(
      None
      if checkpoint.get("information") is None
      else np.asarray(checkpoint["information"], dtype=np.float64)
    ),
  )


def load_map(path: str | Path) -> BathymetryMap | VecchiaMap:
  """Load a fitted map of either format, ready for prediction.

  Both satisfy :class:`~auv_pose.estimation.terrain.DepthMap`, and both keep
  the same ``predict(points, with_std=..., observation_noise=...)`` signature,
  so a caller that only queries depths does not need to know which it has.

  Note:
      An SVGP checkpoint contains a pickled scikit-learn ``StandardScaler``.
      Pickles are not portable across scikit-learn versions -- if this warns
      about a version mismatch, refit with ``experiments/train_map.py`` rather
      than trusting the loaded scaler, since every depth query passes through
      it. A Vecchia checkpoint has no such object.
  """
  path = Path(path)
  with open(path, "rb") as handle:
    checkpoint = pickle.load(handle)

  # A pickle holding something other than a mapping is not a type error to the
  # caller -- it is the same "this is not a map" answer as a dict missing its
  # keys, and callers catch that as a ValueError. Fall through to the
  # missing-keys path so it reports which keys, rather than only the shape.
  if not isinstance(checkpoint, dict):
    checkpoint = {}

  if checkpoint.get("format") == "vecchia":
    return _load_vecchia(path, checkpoint)

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
