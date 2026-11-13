 # Diffuse sky maps can produce low-coverage schedules

## Summary

When scheduling observations for a diffuse sky map, the returned schedule can
cover a relatively small fraction of the sky-map probability. Increasing
`--max-fields` substantially improves the best coverage found in some runs,
but schedules from runs with the same field limit and deadline can differ
between the default scheduler and a prototype that rescales the MILP objective.

The current results do **not** establish that small objective coefficients or
CPLEX numerical precision are the cause. In particular, rescaling does not
consistently improve the objective, and most larger test cases terminate at a
time limit without proving optimality. Candidate-field limits and solver
termination therefore need to be considered when interpreting the results.

## Reproduction

The test input is `fake_gaussian_30.fits`, a Gaussian sky map with a
30-degree standard deviation. The recorded m4opt version is
`4.0.1.dev2+gf794a8a51.d20261013`; the CPLEX log identifies solver version
`22.2.0.0`.

From the `low_proba_issue` directory, the default scheduler can be run with:

1st runs (default scheduler, 6h `deadline` and 50 `max-fields`):
```shell
m4opt schedule fake_gaussian_30.fits schedule_rubin_100fields.ecsv \
  --write-model model_rubin_100fields.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --timelimit 15min --visits 1 --nside 512 \
  --max-fields 50

m4opt schedule fake_gaussian_30.fits schedule_ztf_100fields.ecsv \
  --write-model model_ztf_100fields.lp \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 6h --timelimit 15min --visits 1 --nside 512 \
  --max-fields 50
```

The **scaled** prototype is a standalone copy in `schedule_scaled.py`. The recorded 6-hour `deadline`, 50 `max-fields` scaled runs used:
```shell
python schedule_scaled.py schedule fake_gaussian_30.fits schedule_rubin_scaled_100.ecsv \
  --write-model model_rubin_scaled.lp --timelimit 20min \
  --event-time 2026-04-21T04:00:00 \
  --mission rubin --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --jobs 0 --max-fields 100

python schedule_scaled.py schedule fake_gaussian_30.fits schedule_ztf_scaled_100.ecsv \
  --write-model model_ztf_scaled.lp --timelimit 20min \
  --event-time 2026-04-21T04:00:00 \
  --mission ztf --bandpass r --exptime-min 300s \
  --deadline 6h --visits 1 --nside 512 --jobs 0 --max-fields 100
```

Tests runs with 100 `max-fields`, for the default scheduler:
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
And for the scaled scheduler:
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

Finally the 100 `max fields` test were rerun with a longer `deadline` of 8h, with the default scheduler:
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
```
And for the scaled scheduler:
```bash
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

## Results

Covered probability is the objective value reported by the scheduler.
Observation counts are included for context; they are not themselves the
optimization objective. The 100-field runs below reached a time limit (or were
aborted), so their reported schedules are feasible incumbents, not proven
optima.

| Deadline | Mission | Scheduler | Max fields | Observations | Covered probability | Solver status |
|---|---|---|---:|---:|---:|---|
| 6 h | Rubin | Default | 50 | 43 | 0.0508421138 | Integer optimal solution |
| 6 h | Rubin | Scaled | 50 | 44 | 0.0511976420 | Time limit exceeded |
| 6 h | ZTF | Default | 50 | 50 | 0.2414937921 | Integer optimal solution |
| 6 h | ZTF | Scaled | 50 | 50 | 0.2414937921 | Time limit exceeded |
| 6 h | Rubin | Default | 100 | 56 | 0.0766280504 | Time limit exceeded |
| 6 h | Rubin | Scaled | 100 | 51 | 0.0733376978 | Time limit exceeded |
| 6 h | ZTF | Default | 100 | 68 | 0.3748463590 | Time limit exceeded |
| 6 h | ZTF | Scaled | 100 | 68 | 0.3758029265 | Time limit exceeded |
| 8 h | Rubin | Default | 100 | 52 | 0.07663590027 | Time limit exceeded |
| 8 h | Rubin | Scaled | 100 | 53 | 0.07587719799 | Time limit exceeded |
| 8 h | ZTF | Default | 100 | 87 | 0.39818809744 | Time limit exceeded |
| 8 h | ZTF | Scaled | 100 | 86 | 0.39769014174 | Time limit exceeded |

With the default `--max-fields 50` and a 6-hour deadline, the recorded runs
reported an integer-optimal solution. The results suggest a small benefit for the scaled Rubin run, but no measurable change for ZTF (due to the maximum number of field being reached).

The 100-field results do not show a consistent benefit from scaling. At a
6-hour deadline, the scaled Rubin incumbent has lower coverage than the
default run, while the scaled ZTF incumbent has slightly higher coverage. At
8 hours, both scaled incumbents have slightly lower coverage. Since these
runs did not prove optimality and use time-limited solves, the differences
should not be interpreted as evidence that scaling changes the true optimum. 

Overall, scaling does not produce any significant difference in number of observations or covered probability and is likely not the solution.

## Expected behavior

For a given set of candidate fields and observing constraints, the scheduler
should return a schedule that maximizes the integrated sky-map probability
covered by its observations. When the solver stops before proving optimality,
the incumbent and best bound should be considered together when assessing
schedule quality. However, for a large and diffuse skymap like `fake_gaussian_30.fits`, the scheduler produces plan that stops before the deadline and/or the number of max-fields is reached despite having no blocking constraints.

## Investigation and current hypothesis

The exported unscaled objectives contain many non-zero pixel-probability
coefficients of order `1e-6`. The prototype divides all objective coefficients
by the maximum probability among pixels in the MILP, bringing the largest
coefficient close to one. The exact mathematical optimizer is unchanged by
this positive uniform scaling.

```python
probability_scale = float(np.max(skymap_flat["PROB"], initial=0.0)) or 1.0
```

However, the experiments so far do not confirm that numerical scaling is the
root cause. Increasing `--max-fields` changes the candidate set and improves
the best coverage found in both schedulers. In the 100-field runs, the solver
does not prove optimality, and scaling has no consistent effect on incumbent
coverage. Further controlled tests would be needed to separate numerical
conditioning from candidate truncation, search behavior, and run-to-run
variation.

From the results of ZTF and Rubin, the larger the FOV of a mission is, the less it may be affected. Further test with UVEX and ULTRASAT need to be done to confirm this.

## Available artifacts

- `fake_gaussian_30.fits`: input sky map.
- `schedule_*.ecsv`: generated schedules and solver metadata.
- `rubin*.gif`, `ztf*.gif` and matching PDFs: schedule and coverage visualizations for the tested configurations.


```bash
python \
  -m m4opt._cli.schedule_candidates schedule \
  fake_gaussian_30.fits schedule_rubin_300.ecsv \
  --mission rubin \
  --bandpass r \
  --exptime-min 300s \
  --event-time 2026-04-21T04:00:00 \
  --deadline 6h \
  --visits 1 \
  --nside 512 \
  --max-fields 300 \
  --timelimit 15min \
  --write-candidates candidates_rubin_300.npz
```