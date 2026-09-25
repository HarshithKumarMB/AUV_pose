# AUV_pose

Terrain-aided pose estimation for a BlueROV2 in the
[HoloOcean](https://byu-holoocean.github.io/holoocean-docs/) simulator: survey
the seabed with a multibeam, fit a Vecchia GP map, and smooth the vehicle's
inertial navigation against it.

## Setup

```
nix develop                  # offline scripts and tests
nix run .#sim -- -c "..."    # anything that runs the simulator
nix run .#sim -- -c "python -c \"import holoocean; holoocean.install('Ocean')\""  # once
```

## Pipeline

| Step | Command |
|---|---|
| Survey | `experiments/survey.py --out pass0 --yaw 0` |
| Place soundings | `experiments/georeference.py pass0 --out pass0.csv` |
| Fit the map | `experiments/train_map.py pass*.csv` |
| Fly a test track | `experiments/survey.py --route figure8 --out test0` |
| Score the map | `experiments/score_track.py test0.csv ...` |

## Tests

```
nix develop --command pytest
```
