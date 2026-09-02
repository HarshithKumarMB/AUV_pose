# Figures

Kept deliberately, unlike the regenerable plots the scripts drop at the repo
root — `.gitignore` excludes `*.png` everywhere except here.

## `navigate_runs.png`

Four `navigate.py` runs over the same 300-step (10 s) course, tracing the two
defects found in the first simulator runs and the effect of fixing each.

| run | configuration | final EKF error |
|---|---|---|
| gyro only | no attitude correction, no DVL | 156.9 m |
| + attitude correction | `AttitudeFilter` bounding tilt | 31.1 m |
| + DVL | velocity aiding as well | 3.9 m |
| truth attitude (bound) | `--truth-attitude`, a diagnostic bound | 26.6 m |

Panels: **(a)** position error against ground truth, log scale; **(b)** attitude
error, the cause; **(c)** acceleration fed to the filter minus the truth, the
mechanism — gravity failing to cancel as tilt error grows; **(d)** the resulting
track with both fixes.

**These are not a controlled comparison.** Guidance is closed on the estimate,
so each configuration flies a genuinely different trajectory and experiences
different manoeuvres. Differences between runs indicate estimator quality; they
are not like-for-like. It is why the `+ DVL` run ends with *higher* attitude
error than `+ attitude correction` while having 8x less position error.

Made with the plotting script noted in the commit that added this file, from run
logs produced by `experiments/navigate.py`. The logs are not in the repository —
regenerating the figure means re-running the four configurations:

```
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --gyro-only --no-dvl --out gyro.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --no-dvl        --out corrected.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300                 --out dvl.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --truth-attitude --no-dvl --out bound.csv"
```
