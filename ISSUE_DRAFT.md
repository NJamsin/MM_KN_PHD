# Diffuse sky maps produce unexpectedly low optimized coverage

## Summary

When scheduling observations for a very diffuse sky map (a Gaussian with a
standard deviation of 30 degrees), the resulting integrated sky probability
covered by the schedule appears unexpectedly low. This is a coverage-quality
issue, not a completely empty schedule: the supplied runs do contain
observations but more observations .

The exported MILP models have very small non-zero probability coefficients.
This may indicate a numerical-scaling issue in the objective, but the available
results do not prove that CPLEX precision is the cause: both runs report an
integer-optimal solution, with matching best bounds.

## Reproduction

Using the 30-degree Gaussian sky map and the default Rubin and ZTF missions:

```shell
m4opt schedule fake_gaussian_30.fits schedule_rubin.ecsv \
  --write-model model_rubin.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --max-fields 50

m4opt schedule fake_gaussian_30.fits schedule_ztf.ecsv \
  --write-model model_ztf.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --max-fields 50
```

The recorded m4opt version is `4.0.1.dev2+gf794a8a51.d20261013`.

## Actual results

| Mission | Observations | Covered probability (objective) | Solver status |
|---|---:|---:|---|
| Rubin | 43 | 0.0508421138 | integer optimal solution |
| ZTF | 50 | 0.2414937921 | integer optimal solution |

The exported LP objectives contain 23,173 terms for Rubin and 110,530 for ZTF.
Their non-zero coefficients are approximately `2.33e-6` to `2.54e-6` for Rubin
and `1.77e-6` to `2.54e-6` for ZTF. The reported best bounds match the objective
values to the precision recorded in the ECSV metadata.

The concern is that these schedules capture surprisingly little probability
for the available observing windows. An independently verified feasible
schedule with higher coverage would help determine whether the optimizer is
missing useful solutions or whether the low coverage is explained by the
visibility and candidate-field constraints.

## Expected behavior

The schedule should maximize the integrated posterior probability covered by
the selected observations, subject to the supplied visibility, timing, and
mission constraints. If a feasible schedule covering more probability exists
among the modeled candidates, the optimizer should find it. The expected
coverage should be checked against an independent feasible benchmark rather
than assumed from the map's 90% credible region alone.

## Suspected cause

The pixel probabilities in the objective are on the order of `1e-6`. Objective
coefficients at this scale may interact poorly with solver numerical
scaling/tolerances, potentially making meaningful objective differences harder
to distinguish. This is a hypothesis, not a confirmed diagnosis: the supplied
runs report optimal solutions and matching bounds, so solver logs and an
independent objective check are needed to establish whether numerical scaling
is responsible.

## Possible fix

Apply a single positive scale factor to all probability coefficients before
building the MILP, for example:

```text
scaled_probability[i] = probability[i] / max(probability)
```

This leaves the exact optimizer unchanged while bringing the objective
coefficients to a better numerical scale. Keep user-visible probabilities and
objective values in their original units: convert the solver objective back
when reporting it, and scale any objective cutoff consistently. A global
factor should be applied uniformly; clipping or independently renormalizing
pixels would change the optimization problem.

## Prototype scaling test

A standalone copy of `schedule.py` with this objective scaling is provided as
`schedule_scaled.py`. For this test, it uses the maximum probability among the
pixels in the MILP as the scale factor. The solver objective and cutoff are
scaled internally; objective values, best bounds, and progress values are
converted back to probability units when written.

The original and scaled implementations were run with the same solver settings
(`--jobs 1`) on the included sky map:

| Mission | Unscaled observations | Scaled observations | Unscaled coverage | Scaled coverage |
|---|---:|---:|---:|---:|
| Rubin | 44 | 50 | 0.0508421138 | 0.0511976420 |
| ZTF | 50 | 50 | 0.2414937921 | 0.2414937921 |

Both runs reported an integer-optimal solution. In these runs, scaling raised
the largest non-zero objective coefficient from about `2.54e-6` to about `1.0`.
For Rubin, the scaled run selected six additional fields and increased covered
probability by about `3.56e-4`; ZTF's schedule and coverage were effectively
unchanged. This is evidence that scaling can improve this Rubin result, but it
does not by itself establish the cause or guarantee improvement for other
inputs.

Reproduce the scaled runs from this directory with:

```shell
python schedule_scaled.py schedule fake_gaussian_30.fits schedule_rubin_scaled.ecsv \
  --write-model model_rubin_scaled.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --jobs 0 --max-fields 50

python schedule_scaled.py schedule fake_gaussian_30.fits schedule_ztf_scaled.ecsv \
  --write-model model_ztf_scaled.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --jobs 0 --max-fields 50
```

The corresponding unscaled rerun artifacts are `schedule_rubin_rerun.ecsv`,
`schedule_ztf_rerun.ecsv`, `model_rubin_rerun.lp`, and `model_ztf_rerun.lp`.

**note:** In both cases, the number of observations hit 50, the effective ceiling for these runs, increasing the `max-fields` will lead to different results.

## Increasing `max-fields`
The default scheduler runs where re-run with `--max-fields 100`

Commands:

```shell
m4opt schedule fake_gaussian_30.fits schedule_rubin_100fields.ecsv \
  --write-model model_rubin_100fields.lp \
  --event-time 2026-04-21T04:00:00 --timelimit 15min\
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --max-fields 100

m4opt schedule fake_gaussian_30.fits schedule_ztf_100fields.ecsv \
  --write-model model_ztf_100fields.lp \
  --event-time 2026-04-21T04:00:00 --timelimit 15min \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --max-fields 100
```

The default runs results are vastly differents despite the Rubin (50 `max-fields`) run being marked as integer optimal with 44 fields.

| Mission | Observations | Covered probability (objective) | Gain (from 50 fields)| Solver status |
|---|---:|:---:|:---:|---|
| Rubin | 56 | 0.0766280504 | 0.0257859366 | interrupted run (15min runtime) |
| ZTF | 68 | 0.3748463590 | 0.1333525669 | interrupted run (15min runtime) | 

So the default scheduler can, by itself, partially overcome this issue. However, from `rubin_100fields.gif` we can see that this issue is still present even when increasing the `max-fields` as several observations could have been conducted in the remaining time to increase the coverage. (For ZTF no more observations were possible due to the deadline.)

## Scaling test with higher `max-fields`
Commands: 
```shell
python schedule_scaled.py schedule fake_gaussian_30.fits schedule_rubin_scaled_100.ecsv \
  --write-model model_rubin_scaled.lp --timelimit 15min \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --jobs 0 --max-fields 100

python schedule_scaled.py schedule fake_gaussian_30.fits schedule_ztf_scaled_100.ecsv \
  --write-model model_ztf_scaled.lp --timelimit 20min \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 8h --visits 1 --nside 512 --jobs 0 --max-fields 100
```
| Mission | Unscaled observations | Scaled observations | Unscaled coverage | Scaled coverage |
|---|---:|---:|---:|---:|
| Rubin | 56 | 51 | 0.0766280504 | 0.0733376978 |
| ZTF | 68 | 68 | 0.3748463590 | 0.3758029265 |

The results are not really encouraging for the potential fix as the scaled results with a higher max-fields performs the same for Rubin (ZTF can't go beyond 68 fields with this configuration due to the deadline).

## Trying with a longer deadline:
```bash
m4opt schedule fake_gaussian_30.fits schedule_ztf_long.ecsv \
  --write-model model_ztf_long.lp \
  --event-time 2026-04-21T04:00:00 --timelimit 20min \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 8h --visits 1 --nside 512 --max-fields 100

m4opt schedule fake_gaussian_30.fits schedule_rubin_long.ecsv \
  --write-model model_rubin_long.lp \
  --event-time 2026-04-21T04:00:00 --timelimit 20min \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 8h --visits 1 --nside 512 --max-fields 100

python schedule_scaled.py schedule fake_gaussian_30.fits schedule_ztf_scaled_long.ecsv \
  --write-model model_ztf_scaled.lp --timelimit 20min \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 8h --visits 1 --nside 512 --jobs 0 --max-fields 100

python schedule_scaled.py schedule fake_gaussian_30.fits schedule_rubin_scaled_long.ecsv \
  --write-model model_rubin_scaled.lp --timelimit 20min \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 8h --visits 1 --nside 512 --jobs 0 --max-fields 100
```
| Mission | Unscaled observations | Scaled observations | Unscaled coverage | Scaled coverage |
|---|---:|---:|---:|---:|
| ZTF | 87 | 86 | 0.39818809744 | 0.39769014174 |
| Rubin | 52 | 53 | 0.07663590027 | 0.07587719799 |

Once again, similar results but for the same run time the scaled scheduler performs slightly worse. Moreover, the rubin values are not fully coherent with the shorter run (more observations produced on a shorter deadline)

**Scaling** is likely not the solution unfortunately !

## Available files

- `fake_gaussian_30.fits`: input sky map used in the test.
- `schedule_scaled.py`: standalone copy of the scheduler with objective
  scaling enabled.
- `schedule_rubin.ecsv`, `schedule_ztf.ecsv`: generated schedules and solver
  metadata.
- `model_rubin.lp`, `model_ztf.lp`: exported MILP models.
- `schedule_*_scaled.ecsv`, `model_*_scaled.lp`: scaled runs.
- `rubin.gif/.pdf`, `ztf.gif/.pdf`: schedule/coverage visualizations.
- `rubin_100fields.gif/.pdf`, `ztf_100fields.gif/.pdf`: schedule/coverage visualizations for 100 `max-fields`.
- `rubin_scaled.gif/.pdf`, `ztf_scaled.gif/.pdf`: schedule/coverage visualizations with scaled scheduler.
- `rubin_scaled_100.gif/.pdf`, `ztf_scaled_100.gif/.pdf`: schedule/coverage visualizations with scaled scheduler and 100 `max-fields`.
- `rubin_scaled_long.gif/.pdf`, `ztf_scaled_long.gif/.pdf`: schedule/coverage visualizations with scaled scheduler and 100 `max-fields` and an 8hr `deadline`.
- `rubin_long.gif/.pdf`, `ztf_long.gif/.pdf`: schedule/coverage visualizations with default scheduler and 100 `max-fields` and an 8hr `deadline`.