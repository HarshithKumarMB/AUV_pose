"""Does the fit recover a lengthscale it was *given*?

Every Vecchia fit on this survey has returned a lengthscale that tracks the
decimation cell -- 7.09 m at 0.25 m cells, 18.0 m at 4 m cells, each one
converged. Two explanations survive, and they point opposite ways:

1. **The seabed is not a stationary Matern.** Each decimation presents a
   different effective structure and the best stationary fit honestly tracks
   it. Then the model is what needs work, and the map's uncertainty is
   describing a field it cannot represent.
2. **The pipeline is biased.** Something about maximin ordering, the
   conditioning sets, the linear mean or the optimiser drags the estimate
   toward the sampling scale. Then nothing measured so far means what it says.

No fit to real soundings can separate these, because the seabed's true
lengthscale is unknown -- which is the circularity this script exists to break.
Draw a field from a Matern whose hyperparameters *are* known, **at the real
survey's own positions**, and put it through the same pipeline. If the estimate
comes back near the truth at every cell, the pipeline is sound and the real
seabed is genuinely multi-scale. If it tracks the cell here too, the defect is
ours.

**Generating and fitting must not share a conditioning set.** With the same
``m`` the data is an exact sample from the model being fitted, so recovery
would test the optimiser and nothing else. The draw therefore uses a much
larger ``m`` than the fit -- close enough to an exact Matern draw that the gap
being measured is the fit's.

Run::

    nix develop --command python experiments/check_lengthscale_recovery.py \\
        ~/data/auv_pose/surveys_v2/pass*.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from auv_pose.io.soundings import load_soundings, soundings_to_arrays
from auv_pose.mapping.vecchia import build_structure, draw, fit_vecchia
from experiments.train_map import decimate

#: The field the draw is asked for. Chosen near what the real fits report, so
#: the recovery question is asked in the regime that actually arises rather
#: than at some comfortable round number.
TRUE_LENGTHSCALE = (15.0, 5.0)
TRUE_AMPLITUDE = 50.0
TRUE_NUGGET = 0.7

#: Conditioning size for the draw. Five times the fit's, so the generated field
#: is close to an exact Matern realisation rather than a sample from the very
#: approximation under test.
GENERATING_CONDITIONING = 150


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("surveys", nargs="+", type=Path)
  parser.add_argument(
    "--base-cell",
    type=float,
    default=0.25,
    help="cell the field is drawn at; the coarser tests decimate this",
  )
  parser.add_argument(
    "--cells",
    type=float,
    nargs="+",
    default=(0.25, 1.0, 2.0),
    help="decimation cells to fit at",
  )
  parser.add_argument("--conditioning", type=int, default=30)
  parser.add_argument(
    "--near",
    type=int,
    default=None,
    help="nearest-neighbour share of --conditioning; default all nearest",
  )
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--seed", type=int, default=11)
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="cap the drawn field's size, for a quicker check",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  frame = load_soundings(args.surveys)
  X, _ = soundings_to_arrays(frame)
  print(f"Loaded {len(X)} soundings")

  positions, _ = decimate(X, np.zeros(len(X)), args.base_cell)
  if args.limit is not None and len(positions) > args.limit:
    keep = np.random.default_rng(args.seed).choice(
      len(positions), args.limit, replace=False
    )
    positions = positions[np.sort(keep)]
  print(
    f"Drawing at {len(positions)} positions from the real survey, "
    f"decimated at {args.base_cell} m"
  )

  log_amplitude = torch.tensor(math.log(TRUE_AMPLITUDE), dtype=torch.float64)
  log_lengthscale = torch.log(
    torch.tensor(TRUE_LENGTHSCALE, dtype=torch.float64)
  )
  noise = torch.tensor(TRUE_NUGGET, dtype=torch.float64)

  print(
    f"  truth: lengthscales {TRUE_LENGTHSCALE} m, "
    f"amplitude {TRUE_AMPLITUDE} m^2, nugget {TRUE_NUGGET} m^2"
  )
  print(
    f"  drawing with m={GENERATING_CONDITIONING} (fit uses "
    f"m={args.conditioning}), so the two do not share a model"
  )

  structure = build_structure(
    positions,
    m=GENERATING_CONDITIONING,
    n0=max(GENERATING_CONDITIONING, 64),
  )
  field = draw(
    structure,
    log_amplitude,
    log_lengthscale,
    noise,
    count=1,
    seed=args.seed,
  )
  print(
    f"  drawn: sd {field.std():.2f} m against sqrt(sf^2 + nugget) = "
    f"{math.sqrt(TRUE_AMPLITUDE + TRUE_NUGGET):.2f} m"
  )

  print(
    f"\n{'cell':>6}  {'points':>8}  {'lengthscales':>18}  "
    f"{'amplitude':>10}  {'nugget':>8}"
  )
  results = []
  for cell in args.cells:
    if cell <= args.base_cell:
      sub_x, sub_y = positions, field
    else:
      sub_x, sub_y = decimate(positions, field, cell)

    fitted = fit_vecchia(
      sub_x,
      sub_y,
      m=args.conditioning,
      steps=args.steps,
      device=args.device,
      near=args.near,
    )
    lengthscale = fitted.hyper.lengthscale
    amplitude = fitted.hyper.amplitude
    nugget = fitted.hyper.noise
    results.append((cell, lengthscale))

    print(
      f"{cell:6.2f}  {len(sub_x):8d}  "
      f"{lengthscale[0]:8.2f} x {lengthscale[1]:6.2f}  "
      f"{amplitude:10.2f}  {nugget:8.3f}"
    )

  print("\nVerdict")
  spread = max(r[1][0] for r in results) / min(r[1][0] for r in results)
  error = [
    abs(r[1][0] - TRUE_LENGTHSCALE[0]) / TRUE_LENGTHSCALE[0] for r in results
  ]
  print(f"  lengthscale varies by {spread:.2f}x across the cells")
  print(f"  worst error against the truth: {max(error):.0%}")

  if spread < 1.3 and max(error) < 0.3:
    print(
      "  PASS -- the pipeline recovers a stationary lengthscale at every "
      "cell, so the real survey's drift is the seabed, not the method"
    )
  else:
    print(
      "  FAIL -- the estimate moves with the sampling even on a field that "
      "is stationary by construction. Fix this before reading anything "
      "into a fitted lengthscale"
    )


if __name__ == "__main__":
  main()
