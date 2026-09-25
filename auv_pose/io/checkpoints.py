"""Saving and loading a fitted bathymetry map as a NumPy ``.npz``."""

from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch

from auv_pose.mapping.svgp import BathymetryMap, SVGPModel
from auv_pose.mapping.vecchia import (
  MeanBasis,
  VecchiaHyperparameters,
  VecchiaMap,
  VecchiaStructure,
)

#: Bumped whenever a stored field changes meaning.
VERSION = 1


def save_map(path: str | Path, bathymetry: BathymetryMap | VecchiaMap) -> None:
  """Write either kind of map to ``path``, which should end in ``.npz``."""
  if isinstance(bathymetry, VecchiaMap):
    arrays = _vecchia_arrays(bathymetry)
  else:
    arrays = _svgp_arrays(bathymetry)
  with open(path, "wb") as handle:
    np.savez_compressed(handle, **{"version": np.array(VERSION), **arrays})


def load_map(path: str | Path) -> BathymetryMap | VecchiaMap:
  """Read a map written by :func:`save_map`."""
  with np.load(path, allow_pickle=False) as stored:
    arrays = dict(stored)
  if arrays.get("version") != VERSION or "kind" not in arrays:
    raise ValueError(f"{path} is not a version {VERSION} map checkpoint")
  kind = str(arrays["kind"])
  if kind == "vecchia":
    return _vecchia_map(arrays)
  if kind == "svgp":
    return _svgp_map(arrays)
  raise ValueError(f"{path} holds an unknown map kind {kind!r}")


def _vecchia_arrays(bathymetry: VecchiaMap) -> dict[str, Any]:
  structure, hyper = bathymetry.structure, bathymetry.hyper
  arrays = {
    "kind": np.array("vecchia"),
    "points": structure.points,
    "neighbours": structure.neighbours.astype(np.int32),
    "order": structure.order.astype(np.int32),
    "n0": np.array(structure.n0),
    "residual": bathymetry.residual,
    "beta": bathymetry.beta,
    "noise": bathymetry.noise,
    "log_amplitude": np.array(hyper.log_amplitude),
    "log_lengthscale": np.array(hyper.log_lengthscale),
    "log_noise": np.array(hyper.log_noise),
    "degree": np.array(bathymetry.basis.degree),
    "loglik_trace": np.array(bathymetry.loglik_trace, dtype=float),
  }
  if bathymetry.information is not None:
    arrays["information"] = bathymetry.information
  return arrays


def _vecchia_map(arrays: dict[str, np.ndarray]) -> VecchiaMap:
  return VecchiaMap(
    structure=VecchiaStructure(
      points=arrays["points"],
      neighbours=arrays["neighbours"].astype(np.int64),
      order=arrays["order"].astype(np.int64),
      n0=int(arrays["n0"]),
    ),
    residual=arrays["residual"],
    beta=arrays["beta"],
    hyper=VecchiaHyperparameters(
      log_amplitude=float(arrays["log_amplitude"]),
      log_lengthscale=(
        float(arrays["log_lengthscale"][0]),
        float(arrays["log_lengthscale"][1]),
      ),
      log_noise=float(arrays["log_noise"]),
    ),
    noise=arrays["noise"],
    loglik_trace=arrays["loglik_trace"].tolist(),
    basis=MeanBasis(kind="polynomial", degree=int(arrays["degree"])),
    information=arrays.get("information"),
  )


def _svgp_arrays(bathymetry: BathymetryMap) -> dict[str, Any]:
  # Forced to the CPU: a map evaluated on a GPU holds CUDA tensors.
  arrays = {
    "kind": np.array("svgp"),
    "x_mean": bathymetry.x_mean,
    "x_scale": bathymetry.x_scale,
    "y_mean": np.array(bathymetry.y_mean),
    "y_std": np.array(bathymetry.y_std),
  }
  for prefix, module in (
    ("model", bathymetry.model),
    ("likelihood", bathymetry.likelihood),
  ):
    for key, value in module.state_dict().items():
      arrays[f"{prefix}/{key}"] = value.detach().cpu().numpy()
  return arrays


def _svgp_map(arrays: dict[str, np.ndarray]) -> BathymetryMap:
  def state(prefix: str) -> dict[str, torch.Tensor]:
    return {
      key.removeprefix(f"{prefix}/"): torch.from_numpy(value)
      for key, value in arrays.items()
      if key.startswith(f"{prefix}/")
    }

  model_state = state("model")
  model = SVGPModel(model_state["variational_strategy.inducing_points"])
  model.load_state_dict(model_state)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  likelihood.load_state_dict(state("likelihood"))
  return BathymetryMap(
    model,
    likelihood,
    x_mean=arrays["x_mean"],
    x_scale=arrays["x_scale"],
    y_mean=float(arrays["y_mean"]),
    y_std=float(arrays["y_std"]),
  )
