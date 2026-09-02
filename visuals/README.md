# Figures

Kept deliberately, unlike the regenerable plots the scripts drop at the repo
root — `.gitignore` excludes `*.png` everywhere except here.

## `navigate_runs.png`

Five `navigate.py` runs over the same 300-step (10 s) course, tracing the three
defects found in the simulator runs and the effect of fixing each.

| run | configuration | final EKF error | velocity error |
|---|---|---|---|
| gyro only | no attitude correction, no DVL | 27.6 m | 3.86 m/s |
| + attitude correction | `AttitudeFilter` bounding tilt | 14.6 m | 1.69 m/s |
| + DVL, wrong axes | velocity aiding, DVL frame uncorrected | 31.5 m | 4.37 m/s |
| + DVL, axes corrected | `DVL_AXES` applied | **0.68 m** | 0.07 m/s |
| truth attitude (bound) | `--truth-attitude`, a diagnostic bound | 13.8 m | 1.47 m/s |

The three defects, in the order they were found:

1. **Attitude was propagated from the gyro alone.** Bounding tilt against the
   accelerometer takes the run to 14.6 m — which is the truth-attitude bound of
   13.8 m, so attitude is no longer what limits it.
2. **The DVL's axes are not the `OrientationSensor`'s.** Its y and z are
   inverted, so before `DVL_AXES` the DVL was *worse than not having one*:
   31.5 m against 14.6 m. The filter was handed a systematically wrong velocity
   and told to trust it at 0.1 m/s.
3. **The IMU bias sigmas were misread as bias standard deviations.** HoloOcean
   applies `AccelBiasSigma`/`AngVelBiasSigma` as **per-sample random-walk
   increments**, so bias grows as `sigma·sqrt(n)`. At the previous value of
   0.01 — measured with `ReturnBias` against a motionless vehicle — the gyro
   bias reached 14 °/s and the accelerometer bias 0.35 m/s² after 300 samples,
   around 200× a real MEMS unit. Size them backwards from a target bias:
   `sigma = bias_at_n / sqrt(n)`.

That third defect dominated the other two, and its discovery changed what the
first two mean. In the previous version of this figure the story was gravity
leaking into acceleration through a diverging attitude; with a correctly
specified gyro, the spurious-acceleration panel is flat at zero for every run
and that mechanism is gone. Panel **(b)** now carries the mechanism instead:
without a DVL, velocity is unobservable and plateaus near 1.5 m/s regardless of
how good attitude is; with a mis-framed DVL it grows without bound; with a
correct one it converges to 0.07 m/s.

Panels: **(a)** position error against ground truth, log scale; **(b)** velocity
error, the mechanism; **(c)** attitude error over the vehicle's own rotation;
**(d)** the resulting track with all three fixed.

**Read panel (c) carefully.** Attitude error and the vehicle's physical rotation
are plotted together because otherwise the panel reads as if the attitude filter
failed. It did not: attitude holds to 0.5° in every run until the vehicle itself
starts rotating, at around t = 7 s in the two runs whose position estimate
diverged first. The controller chased a bad estimate until it tumbled the
vehicle. The attitude error there is a *consequence* of divergence, not its
cause — the reverse of what the earlier version of this figure showed.

**These are not a controlled comparison.** Guidance is closed on the estimate,
so each configuration flies a genuinely different trajectory and experiences
different manoeuvres. Differences between runs indicate estimator quality; they
are not like-for-like.

Regenerating the figure means re-running the five configurations. The logs are
not in the repository:

```
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --gyro-only --no-dvl      --out gyro.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --no-dvl                  --out att.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --raw-dvl-axes            --out dvlraw.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300                           --out dvl.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --truth-attitude --no-dvl --out truth.csv"
```

`--accel-bias-sigma 0.01 --gyro-bias-sigma 0.01` reproduces the runs from before
the IMU was measured.
