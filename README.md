# AUV_pose

Terrain-aided pose estimation for an underwater vehicle (BlueROV2) in the
[HoloOcean](https://byu-holoocean.github.io/holoocean-docs/) simulator.

IMU dead reckoning drifts. The approach here corrects it against a bathymetry map:
survey the seabed with a downward multibeam sonar, fit a Vecchia-approximated
Gaussian process to the soundings, and condition an on-manifold unscented
smoother on it.

The smoother and its building blocks are in `auv_pose/estimation/` but not yet
driven by an experiment: `navigate.py` still runs the earlier EKF, with the
map's depth as a measurement of `z`.

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
| 1. Survey the seabed | `python experiments/survey.py --out pass0.csv` | yes | `pass0.csv` — `x, y, z` |
| 2. Fit the bathymetry map | `python experiments/train_map.py pass0.csv` | no | `vecchia_bathymetry.pkl`, `gp_bathymetry_surface.png` |
| 3. Query a depth | `python experiments/predict_depth.py` | no | prints depth |
| 4. Navigate | `python experiments/navigate.py` | yes | `navigation.csv` |
| 5. Plot tracks and error | `python experiments/plot_trajectory.py` | no | `trajectory_*.png` |

**No survey is committed.** The current four-heading survey lives on the data
volume at `~/data/auv_pose/surveys_v2/pass{0,45,90,135}.csv`, 174 MB; see
`survey.py --help` for flying another. `train_map.py`'s defaults are the
configuration the map ships in, scored on an 8 m blocked holdout. Every script
takes `--help`.

`survey.py` and `train_map.py` refuse to overwrite an existing output, since a
survey costs a simulator run to reproduce; pass `--out` to write elsewhere, or
`--force` once you are sure.

### Diagnosing the sonar

- `check_multibeam.py` captures raw pings over a chosen patch;
  `check_beam_validity.py` scores them against a ray-cast through the octree.
- `capture_scene.py` photographs a patch of seabed. This is what settled a
  three-session disagreement between sonar and octree — see below.

### Other experiments

- `sonar_survey.py` — live sonar viewer, imaging or sidescan, with optional IMU
  dead reckoning. For inspecting raw returns; not part of the pipeline.

## Layout

```
auv_pose/                    # algorithms -- importable, no I/O, no simulator
  estimation/                # state manifold, unscented transform, IMU
                             #   propagation, RTS smoothers; EKF + strapdown
  mapping/                   # Vecchia GP (kernels, ordering, vecchia), SVGP
                             #   baseline, sonar ranges, octree + raycast
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
nix develop --command pytest      # 477 tests, none need the simulator
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

Survey CSVs store `x, y, z`: where a beam struck the seabed, in the world frame,
`z` increasing upward. The GP models that elevation directly.
`auv_pose.io.soundings` owns the schema.

The simulator caches an octree of the world as JSON on disk, so the seabed can be
read without a run — `auv_pose.mapping.octree`, traced with
`auv_pose.mapping.raycast`.

**It is a reference, not ground truth.** Over bare seabed it and the multibeam
agree to a 0.035 m MAD-std, below the sonar's 0.0996 m quantisation. But it holds
only landscape: the pipelines lying on the Dam seabed are absent from it, so the
sonar reads 4-5 m short over them and is right to. Doubling `octree_max` leaves
the returns bit-identical, so the sonar does not consult it at all. Score with
`experiments/check_beam_validity.py`, and read a disagreement as a question about
which of the two is incomplete.

`vendor/holoocean/` is an unmodified copy of upstream tag `v2.3.0`, reduced to the
596 KB the build needs. Upstream is a private, Epic-gated repository —
`vendor/holoocean/VENDOR.md` explains the provenance and why this is a copy rather than
a submodule.
