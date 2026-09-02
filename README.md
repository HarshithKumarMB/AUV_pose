# AUV_pose

Terrain-aided pose estimation for an underwater vehicle (BlueROV2) in the
[HoloOcean](https://byu-holoocean.github.io/holoocean-docs/) simulator.

IMU dead reckoning drifts. The approach here corrects it against a bathymetry map: survey
the seabed with a downward singlebeam sonar, fit a sparse Gaussian process to the
soundings, then use `sonar_altitude + map_depth` as a position measurement in an EKF.

## Setup

Requires [Nix](https://nixos.org/) with flakes enabled.

```
nix develop           # Python env for the offline scripts
nix develop .#sim     # adds the OpenGL/Vulkan/X11 stack the Unreal binary needs
```

Both set `HOLODECKPATH` to `./.holoocean` (gitignored) and put the vendored HoloOcean
client on `PYTHONPATH`.

`nix develop .#sim` is interactive only — it drops you into an FHS environment, and a
`--command` passed to it is silently discarded. To run something non-interactively under
the simulator environment, use the package form:

```
nix run .#sim -- -c "python experiments/navigate.py"
```

The simulator worlds are a **5.2 GB** download, not included. Once:

```
nix run .#sim -- -c "python -c \"import holoocean; holoocean.install('Ocean')\""
```

## Pipeline

Run from the repository root.

| Step | Command | Needs sim | Output |
|---|---|---|---|
| 1. Survey the seabed | `python experiments/survey.py` | yes | `map1.csv` — `x, y, sonar_depth` |
| 2. Fit the GP bathymetry map | `python experiments/train_map.py` | no | `svgp_bathymetry.pkl`, `gp_bathymetry_surface.png` |
| 3. Query a depth | `python experiments/predict_depth.py` | no | prints depth |
| 4. Navigate with the EKF | `python experiments/navigate.py` | yes | `wp_c.csv` |
| 5. Plot tracks and error | `python experiments/plot_trajectory.py` | no | `trajectory_*.png` |

Steps 2, 3 and 5 need no simulator: `map.csv` and `map1.csv` are committed (~82 k
soundings). Every script takes `--help`.

### Other experiments

- `sonar_survey.py` — live sonar viewer, imaging or sidescan, with optional IMU
  dead reckoning. For inspecting raw returns; not part of the pipeline.

## Layout

```
auv_pose/                    # algorithms -- importable, no I/O, no simulator
  estimation/                # quaternion, strapdown, filters, smoothers
  mapping/                   # SVGP bathymetry, sonar range extraction
  io/                        # soundings, checkpoints, run logs
experiments/                 # runnable scripts composing auv_pose
tests/                       # pytest; no simulator required
flake.nix                    # Python env + FHS shell for the simulator
scripts/update-vendor.sh     # verify the vendored client against upstream
vendor/holoocean/            # HoloOcean 2.3.0 Python client (MIT), see VENDOR.md
```

`auv_pose` holds algorithms; anything that opens a file, draws a plot or talks to
HoloOcean lives in `experiments/`.

## Tests

```
nix develop --command pytest      # 109 tests, none need the simulator
nix flake check                   # the same suite, in a sandbox
```

Note that `experiments/` is excluded from collection: several of its scripts open a
world at import time, so collecting them would launch the simulator.

## Conventions

`auv_pose/estimation/` is organised by causality: `filters.py` holds causal
recursive estimators and `smoothers.py` the non-causal backward pass. State is a
value passed through `predict(state, ...)` and `condition(state, obs)`, so a run's
history is a list of values and smoothing is a pure function over it.

World frame is **NED** — x north, y east, **z down** — matching HoloOcean's sensors
in the `IMUSocket`, with gravity `[0, 0, +9.81]`. Quaternions are scalar-first
`[w, x, y, z]` and rotate body vectors into the world. See `auv_pose/estimation/`.

Survey CSVs store `sonar_depth` as a positive range to the seabed; the GP is fitted
on negated depth so the modelled surface increases upward. `auv_pose.io.soundings`
owns that sign flip.

`vendor/holoocean/` is an unmodified copy of upstream tag `v2.3.0`, reduced to the
596 KB the build needs. Upstream is a private, Epic-gated repository —
`vendor/holoocean/VENDOR.md` explains the provenance and why this is a copy rather than
a submodule.
