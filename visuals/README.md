# Figures

Kept deliberately, unlike the regenerable plots the scripts drop at the repo
root — `.gitignore` excludes `*.png` everywhere except here.

## `frame_mismatch.png`

**The defect the rest of this branch was chasing.** holoocean reports the
PoseSensor and OrientationSensor in the frame of the socket they sit in -- NED
in `IMUSocket`, NWU in the default COM socket (`sensors.py:166`). Our config
omitted the socket for both, so ground truth came back in a z-up frame while the
IMU, DVL and depth sensor reported in a z-down one. Gravity was then taken as
z-down (`[0, 0, +9.81]`) in a z-up world.

Those two errors cancel exactly at rest, which is why nothing caught them: an
accelerometer read in a z-down body frame cancels a z-down gravity vector, so
the vehicle does not fall out of the sky and no resting test fails. What happens
instead is that every recovered acceleration comes out mirrored in y and z.

| | before | after |
|---|---|---|
| recovered acceleration vs truth | rms 1.385 m/s², y and z **anticorrelated** | rms **0.098** m/s², all axes +0.99 |
| DVL vs true body velocity | rms 1.298 m/s, needed a `DVL_AXES` sign flip | rms **0.058** m/s, no flip at all |

`DVL_AXES` was never a DVL quirk -- it was a local patch for one symptom of this
mismatch, and it is deleted now that the frames agree.

Both configurations here fly the **same open-loop command sequence**, so this is
a controlled comparison. Regenerate with the probe script noted in the commit
that added this file; `experiments/navigate.py --legacy-frames` reproduces the
old configuration end to end.

**Why this is not shown as a position-error comparison.** Guidance closes on the
estimate, so each configuration flies a genuinely different trajectory, and a
single before/after pair of `navigate.py` runs cannot support a claim about
position error. None is made here. (When this was written dead reckoning alone
spanned 3.7-9.7 m across runs with identical sensors, which made the point
loudly; it is 0.54-0.75 m over three runs now, so the point is quieter but the
logic is unchanged.)

## `navigate_runs.png`

Where `navigate.py` stands on the current code. One 300-step (10 s) run per
configuration, with the previous version of this figure alongside -- the
dead-reckoning column is what changed and why.

| run | EKF | dead reckoning | velocity error | mean attitude error |
|---|---|---|---|---|
| all fixed | 0.51 → **0.24 m** | 7.49 → **0.62 m** | 0.05 → 0.03 m/s | 0.94 → 0.23° |
| mirrored frames (before) | 0.68 → 0.65 m | 9.01 → 1.47 m | 0.03 → 0.03 m/s | 1.03 → 1.50° |
| no DVL | 8.88 → 8.83 m | 9.68 → 8.83 m | 0.95 → 1.41 m/s | 0.75 → 0.92° |
| truth attitude (bound) | 0.27 → 0.25 m | 3.68 → **0.48 m** | 0.01 → 0.03 m/s | 0.05 → 0.08° |

**Dead reckoning was 7.5 m because of two defects, neither of them accelerometer
noise.** The figure it replaces said the drift was residual tilt plus doubly
integrated noise. It was not. Splitting the error by axis separates two causes
that had nothing to do with each other:

*Vertical, 3.9 m: the run started from an assumed rest.* The vehicle is dropped
in negatively buoyant and is still sinking at 0.39 m/s when the first tick
returns, while the strapdown and the EKF both initialised at zero velocity.
Nothing observes velocity in the open loop, so that 0.39 m/s never washed out:
0.39 × 10 s = 3.9 m, and the measured vertical error was +3.93, +3.90 and +3.64 m
in the three runs with correct frames (−4.04 m under `--legacy-frames`, mirrored
like everything else there). It was identical under `--truth-attitude`, which is
why the 3.68 m "bound" in the old table was not a bound on anything -- it was
this, and attitude accounted for 0.58 m of it. Fixed by ticking 60 steps under no
thrust to let the transient decay, then initialising both estimators from the
DVL.

*Horizontal, 6.4 m: the attitude filter was cancelling the acceleration the
strapdown was integrating.* An accelerometer measures specific force, so under
horizontal acceleration `a_h` the reading tilts off `−g` by `atan(a_h / g)`.
Averaged over the run that tilt is 0.95°, and the logged mean attitude error was
0.94° -- the complementary filter was tracking the manoeuvre almost exactly. The
strapdown then rotated the same reading by that tilted attitude and added `g`
back, which cancels most of the acceleration out again. Regressing the recovered
horizontal acceleration on the twice-differentiated truth gives the size of it:

| | slope, before | after |
|---|---|---|
| all fixed | 0.66, 0.64 | 0.94, 0.93 |
| truth attitude (bound) | 1.02, 0.99 | 0.96, 0.98 |
| no DVL | 0.66, 0.68 | 0.59, 0.53 |

A third of the acceleration was being rotated away, and the error was
antiparallel to the true acceleration (median projection −0.86). **That is why
the dead-reckoned track came out short rather than randomly displaced** -- it is
negative feedback between two components, not noise. Fixed by differentiating the
DVL for the vehicle's own acceleration and subtracting it before the reading is
used as a gravity reference.

**`gravity_trust` was supposed to prevent exactly this and structurally cannot.**
A horizontal component adds to `g` in quadrature, so at `a_h = 0.5 m/s²` the
magnitude moves 0.13% and the trust stays at 0.99 -- over the run it averaged
0.95, with 85% of samples above 0.9. A magnitude test detects acceleration
*along* gravity, the one direction that induces no tilt at all. The existing test
for it used 20 m/s², which is the case it can see.

**The `--no-dvl` row is unfixed in the horizontal and that is not an oversight.**
There is nothing else on this vehicle that measures its own acceleration, so the
compensation has no input. Its slope is still 0.59 and its 8.8 m is what the
figure now shows as the weak point. Worth knowing: with the correctly specified
gyro, `--gyro-only` -- no accelerometer correction at all -- held 0.27° over 10 s
against the filter's 0.94°, so on this timescale the uncompensated correction was
a net harm.

**The `mirrored frames` row gets worse in attitude, 1.03 → 1.50°, and that is
consistent.** Under `--legacy-frames` the DVL is sign-flipped along with
everything else, so the compensation is computed from a mirrored velocity and
partly fights the filter instead of helping it. Its horizontal slope shows the
same thing: 0.94 in x but −0.26 in y, where a correctly framed run gets 0.94 in
both. That is the frame mismatch, not a regression in the fix.

**Read the small differences as noise.** Three runs at identical settings gave
0.54, 0.75 and 0.62 m of dead-reckoning error, so the gap between "all fixed" and
"mirrored frames" is not by itself evidence that the frame fix improved position
error -- the controlled evidence for that is `frame_mismatch.png`, where both
configurations fly the same open-loop commands. The `no DVL` velocity error moved
from 0.95 to 1.41 m/s between figures, which is a different trajectory's value
for an unobservable quantity and means nothing.

Panels: **(a)** EKF position error, log scale; **(b)** dead reckoning, log scale,
now within a factor of 1.3 of the truth-attitude bound; **(c)** velocity error,
which is what the DVL observes directly; **(d)** the fully-fixed track, where
dead reckoning now follows the truth instead of stopping 2 m along a 7 m course.

Regenerate with:

```
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300                     --out runs/dvl.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --legacy-frames     --out runs/legacy.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --no-dvl            --out runs/att.csv"
nix run .#sim -- -c "python -u experiments/navigate.py --headless --max-steps 300 --truth-attitude    --out runs/truthatt.csv"
python scripts/plot_runs.py --logs runs/
```

`--accel-bias-sigma 0.01 --gyro-bias-sigma 0.01` reproduces the IMU as it was
configured before it was measured, and `--gyro-only` drops the attitude
correction. Neither defect above has a flag that restores it -- both fixes are
terms in the estimator rather than switches -- so the "before" column comes from
the commit that this figure replaced.
