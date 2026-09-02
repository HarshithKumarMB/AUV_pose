# Figures

Kept deliberately, unlike the regenerable plots the scripts drop at the repo
root — `.gitignore` excludes `*.png` everywhere except here.

## `navigate_runs.png`

Five `navigate.py` runs over the same 300-step (10 s) course, tracing the two
defects found in the first simulator runs and the effect of fixing each.

| run | configuration | final EKF error |
|---|---|---|
| gyro only | no attitude correction, no DVL | 156.9 m |
| + attitude correction | `AttitudeFilter` bounding tilt | 31.1 m |
| + DVL, wrong axes | velocity aiding, DVL frame uncorrected | 31.7 m |
| + DVL, axes corrected | `DVL_AXES` applied | 3.9 m |
| truth attitude (bound) | `--truth-attitude`, a diagnostic bound | 26.6 m |

The third row is the one worth dwelling on: adding the DVL bought **nothing**
— 31.7 m against 31.1 m without it — until its axes were corrected. Its y and z
are inverted relative to the frame the `OrientationSensor` reports, so the
filter was handed a systematically wrong velocity and told to trust it at
0.1 m/s. In panel (a) that run sits directly on top of the one without a DVL.

Panels: **(a)** position error against ground truth, log scale; **(b)** attitude
error, the cause; **(c)** acceleration fed to the filter minus the truth, the
mechanism — gravity failing to cancel as tilt error grows; **(d)** the resulting
track with all fixes.

**These are not a controlled comparison.** Guidance is closed on the estimate,
so each configuration flies a genuinely different trajectory and experiences
different manoeuvres. Differences between runs indicate estimator quality; they
are not like-for-like. It is why the corrected-DVL run ends with *higher*
attitude error than `+ attitude correction` while having 8x less position error.

Made with the plotting script noted in the commit that added this file, from run
logs produced by `experiments/navigate.py`. The logs are not in the repository —
regenerating the figure means re-running the four configurations:

```
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --gyro-only --no-dvl --out gyro.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --no-dvl        --out corrected.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300                 --out dvl.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --truth-attitude --no-dvl --out bound.csv"
```
