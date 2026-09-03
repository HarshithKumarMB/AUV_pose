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

Both put the vendored HoloOcean client on `PYTHONPATH` and set `HOLODECKPATH` to
`~/data/holoocean`, since the world binaries are 5.2 GB. Export `HOLODECKPATH`
before entering the shell to put them elsewhere.

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

Then check the simulator actually launches, before running anything else:

```
nix run .#sim -- -c "python experiments/smoke.py"
nix run .#sim -- -c "python experiments/smoke.py --headless"   # no display
```

`smoke.py` builds a scenario, starts the binary, ticks it and reports every
sensor with its shape. It separates "the binary will not start" from "the
estimator is wrong", which are otherwise easy to confuse. All the simulator
scripts take `--headless`, which passes `-RenderOffScreen`.

## Pipeline

Run from the repository root.

| Step | Command | Needs sim | Output |
|---|---|---|---|
| 0. Extract the true seabed | `python experiments/extract_seabed.py` | no | `seabed_truth.csv` — `x, y, z` |
| 1. Survey the seabed | `python experiments/survey.py` | yes | `map1.csv` — `x, y, z` |
| 2. Fit the GP bathymetry map | `python experiments/train_map.py` | no | `svgp_bathymetry.pkl`, `gp_bathymetry_surface.png` |
| 3. Query a depth | `python experiments/predict_depth.py` | no | prints depth |
| 4. Navigate with the EKF | `python experiments/navigate.py` | yes | `wp_c.csv` |
| 5. Plot tracks and error | `python experiments/plot_trajectory.py` | no | `trajectory_*.png` |

Steps 2, 3 and 5 need no simulator: `map.csv` and `map1.csv` are committed (~82 k
soundings). Every script takes `--help`.

`survey.py` and `train_map.py` refuse to overwrite an existing output. The
committed `map*.csv` and `svgp_bathymetry.pkl` are the only record of a survey
that costs a simulator run to reproduce, so pass `--out` to write elsewhere, or
`--force` once you are sure.

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
nix develop --command pytest      # 135 tests, none need the simulator
nix flake check                   # the same suite, in a sandbox
```

Note that `experiments/` is excluded from collection: several of its scripts open a
world at import time, so collecting them would launch the simulator.

## Conventions

`auv_pose/estimation/` is organised by causality: `filters.py` holds causal
recursive estimators and `smoothers.py` the non-causal backward pass. State is a
value passed through `predict(state, ...)` and `condition(state, obs)`, so a run's
history is a list of values and smoothing is a pure function over it.

World frame is **z up**, with gravity `[0, 0, -9.81]` — `GRAVITY_NWU`, which is
what `navigate.py` uses by default. The vehicle floats at `z ≈ -0.4` and the
seabed sits near `z = -69`. Body axes are HoloOcean's `IMUSocket`, where the
body-to-world rotation is `diag(1, -1, -1)` at rest, so body `+z` points down.
Quaternions are scalar-first `[w, x, y, z]` and rotate body vectors into the
world. See `auv_pose/estimation/`.

(This paragraph previously said NED with `z` down and gravity `[0, 0, +9.81]`.
That was left over from before the frame fix on this branch and contradicted
both the logs and `navigate.py`'s default; `--legacy-frames` still reproduces
the old behaviour.)

Survey CSVs store `x, y, z`: where a beam struck the seabed, in the world frame,
`z` increasing upward. The GP models that elevation directly.
`auv_pose.io.soundings` owns the schema.

The simulator caches the octree its sonar raycasts against as JSON on disk, so
the true seabed can be read without a run — `auv_pose.mapping.octree`. It is
ground truth rather than a second estimate, which is what makes it worth
scoring against: measured through it, the singlebeam's strongest-return range is
biased 4.17 m and a constant beats every bin-selection rule. That is why the
survey flies a multibeam.

`vendor/holoocean/` is an unmodified copy of upstream tag `v2.3.0`, reduced to the
596 KB the build needs. Upstream is a private, Epic-gated repository —
`vendor/holoocean/VENDOR.md` explains the provenance and why this is a copy rather than
a submodule.
