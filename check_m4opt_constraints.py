#!/usr/bin/env python
"""Check whether m4opt schedules violate their observing constraints.

EVERYTHING HERE IS EVALUATED WITH m4opt's OWN CONSTRAINT OBJECTS: the mission
defaults plus any repeated ``--constraint`` options recorded in the schedule's
command metadata. There is no re-implementation of airmass, Sun, Moon, HA, Dec...

How m4opt itself applies the constraints (read from m4opt/_cli/schedule.py):
  1. it evaluates ``mission.constraints(observer_location, field_center, t)``
     on a time grid  t_k = event_time + delay + k * time_step,
  2. it groups the grid points where the constraint is True into closed
     intervals [t_begin, t_end],
  3. it forces every exposure to lie inside one such interval.
So the contract the scheduler can actually guarantee is:
     "all grid points bracketing the exposure satisfy the constraint".
Between two grid points the constraint is NOT evaluated.

Two levels of verdict per schedule:
  FAIL        a grid point bracketing an exposure violates the constraint (or the
              exposure lies outside the grid) -> the schedule breaks m4opt's own
              contract. This is the one that would be a real bug.
  DRIFT-ONLY  contract respected, but at the exact start / middle / end of an
              exposure the constraint is False. This is a consequence of the
              time_step discretisation (the target crossed a limit between two
              grid points), not a bug. Reduce --time-step to shrink it.
  PASS        everything satisfied, also in continuous time.

Usage:
    python check_m4opt_constraints.py schedule_T*.ecsv
    python check_m4opt_constraints.py schedule_X.ecsv --mission mypkg.missions:toy_north
    python check_m4opt_constraints.py *.ecsv --skymap-dir . --no-plots
Exit code 1 if at least one schedule is FAIL.
"""
import argparse
import dataclasses
import importlib
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, get_body, get_sun
from astropy.table import Table
from astropy.time import Time
from astropy.utils.masked import Masked

import m4opt.missions as m4_missions
from m4opt.constraints import (
    AirmassConstraint,
    MoonSeparationConstraint,
    LogicalAndConstraint,
    LogicalNotConstraint,
    LogicalOrConstraint,
    GalacticLatitudeConstraint,
    AvoidSkyRegionConstraint,
)
from m4opt._cli.schedule import _parse_constraint_spec

TOL_S = 0.05  # seconds, numerical tolerance when snapping times to the grid


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def as_bool(x, shape):
    """m4opt constraint output (possibly astropy Masked) -> plain bool array."""
    if isinstance(x, Masked):
        x = x.filled(False)
    return np.broadcast_to(np.asarray(x, dtype=bool), shape).copy()


def as_quantity(x, unit=None):
    q = x if isinstance(x, u.Quantity) else u.Quantity(x)
    return q.to(unit) if unit is not None else q


def flatten_and(c):
    """Top-level operands of an AND tree (m4opt's own objects)."""
    if isinstance(c, LogicalAndConstraint):
        return [leaf for op in c._operands for leaf in flatten_and(op)]
    return [c]


def describe(c):
    """Short human-readable description of an m4opt constraint object."""
    if isinstance(c, (LogicalAndConstraint, LogicalOrConstraint)):
        sep = " & " if isinstance(c, LogicalAndConstraint) else " | "
        return "(" + sep.join(describe(o) for o in c._operands) + ")"
    if isinstance(c, LogicalNotConstraint):
        return "~" + describe(c._operand)
    name = type(c).__name__.replace("Constraint", "")
    items = {}
    if dataclasses.is_dataclass(c):
        items = {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
    else:
        items = {k: v for k, v in vars(c).items() if not k.startswith("_")}
    parts = []
    for k, v in items.items():
        try:
            q = as_quantity(v)
            if q.unit.physical_type == "angle":
                parts.append(f"{k}={q.to_value(u.deg):.4g}deg")
            else:
                parts.append(f"{k}={q:.4g}")
        except Exception:  # noqa: BLE001
            parts.append(f"{k}={v}")
    return f"{name}[{', '.join(parts)}]" if parts else name


def short(s, n=46):
    return s if len(s) <= n else s[: n - 1] + "…"


def load_mission(name, override):
    if override:
        mod, _, attr = override.partition(":")
        return getattr(importlib.import_module(mod), attr)
    if hasattr(m4_missions, name):
        return getattr(m4_missions, name)
    sys.exit(f"Mission '{name}' not found in m4opt.missions; pass --mission module:attr")


def meta_get(meta, key, default=None):
    args = meta.get("args", {}) or {}
    return args.get(key, default)


class Evaluator:
    """Evaluate the mission and schedule-specific constraints on arrays."""

    def __init__(self, mission, additional_constraints=()):
        self.mission = mission
        self.constraint = mission.constraints
        for constraint in additional_constraints:
            self.constraint &= constraint
        self.operands = flatten_and(self.constraint)
        self.labels = [describe(op) for op in self.operands]

    def __call__(self, coords, times):
        loc = self.mission.observer_location(times)
        shape = times.shape
        full = as_bool(self.constraint(loc, coords, times), shape)
        each = np.array(
            [as_bool(op(loc, coords, times), shape) for op in self.operands]
        )  # (n_operands, N)
        return full, each


def additional_constraints_from_command(command):
    """Rebuild supplementary constraints from the saved m4opt command."""
    if not command:
        return []

    tokens = shlex.split(command)
    specs = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--constraint":
            i += 1
            if i == len(tokens):
                raise ValueError("schedule command ends with --constraint but no value")
            specs.append(tokens[i])
        elif token.startswith("--constraint="):
            specs.append(token.partition("=")[2])
        i += 1

    return [_parse_constraint_spec(spec) for spec in specs]


# ----------------------------------------------------------------------------
# Core check
# ----------------------------------------------------------------------------
def check_schedule(path, args):
    path = Path(path)
    name = path.stem.replace("schedule_", "")
    tab = Table.read(path)
    meta = tab.meta
    print(f"\n[{path.name}] columns: {tab.colnames} | {len(tab)} rows")

    mission_name = str(meta_get(meta, "mission", args.mission_name or "ztf"))
    mission = load_mission(mission_name, args.mission)
    additional_constraints = additional_constraints_from_command(meta.get("command"))
    ev = Evaluator(mission, additional_constraints)
    print(
        f"   mission: {mission_name} | {len(ev.operands)} constraints "
        f"({len(additional_constraints)} supplementary):"
    )
    for lab in ev.labels:
        print(f"     - {lab}")

    act = np.char.lower(np.asarray(tab["action"]).astype(str))
    obs = tab[act == "observe"]
    res = dict(test=name, mission=mission_name, n_obs=len(obs))
    if len(obs) == 0:
        print("   => no observation in this schedule: no violation possible")
        res["verdict"] = "EMPTY"
        return res, None

    obs = obs[np.argsort(Time(obs["start_time"]).jd)]
    start = Time(obs["start_time"])
    dur = as_quantity(obs["duration"], u.s)
    end = start + dur
    mid = start + 0.5 * dur
    coords = obs["target_coord"]
    n = len(obs)

    # --- Reconstruct the scheduler's time grid from the file metadata --------
    ts = as_quantity(meta_get(meta, "time_step", 1 * u.min), u.s).value
    delay = as_quantity(meta_get(meta, "delay", 0 * u.s), u.s)
    deadline = meta_get(meta, "deadline", None)
    event_time = meta_get(meta, "event_time", None)
    if event_time is None:
        sys.exit(f"{path.name}: no event_time in metadata, cannot rebuild the grid")
    t0 = Time(event_time) + delay
    n_grid = None
    if deadline is not None:
        n_grid = len(np.arange(delay.value, (as_quantity(deadline, u.s) + ts * u.s).value, ts))
    print(f"   grid: event_time={Time(event_time).isot[:19]} delay={delay:.0f} "
          f"time_step={ts:.0f} s" + (f" ({n_grid} points)" if n_grid else ""))

    rel_start = (start - t0).to_value(u.s)
    rel_end = (end - t0).to_value(u.s)
    k_lo = np.floor((rel_start + TOL_S) / ts).astype(int)
    k_hi = np.ceil((rel_end - TOL_S) / ts).astype(int)
    outside = (k_lo < 0) | (False if n_grid is None else (k_hi > n_grid - 1))
    k_lo_c = np.clip(k_lo, 0, None)
    k_hi_c = k_hi if n_grid is None else np.minimum(k_hi, n_grid - 1)

    # --- (A) contract: every bracketing grid point must satisfy the constraint
    rows, tgrid = [], []
    for i in range(n):
        ks = np.arange(k_lo_c[i], max(k_hi_c[i], k_lo_c[i]) + 1)
        rows.append(np.full(ks.size, i))
        tgrid.append(ks)
    rows = np.concatenate(rows)
    kk = np.concatenate(tgrid)
    t_grid = t0 + kk * ts * u.s
    full_g, each_g = ev(coords[rows], t_grid)

    n_ops = len(ev.operands)
    contract_full = np.ones(n, bool)
    np.logical_and.at(contract_full, rows, full_g)
    frac_ops = np.zeros((n_ops, n))
    cnt = np.bincount(rows, minlength=n).astype(float)
    for j in range(n_ops):
        frac_ops[j] = np.bincount(rows, weights=each_g[j], minlength=n) / cnt
    contract_ops = frac_ops >= 1.0 - 1e-9
    frac_full = np.bincount(rows, weights=full_g, minlength=n) / cnt
    contract_bad = ~contract_full | outside

    # --- (B) exact start / middle / end ---------------------------------------
    exact = {}
    for lab, t in (("start", start), ("middle", mid), ("end", end)):
        exact[lab] = ev(coords, t)  # (full, each)

    n_fail = int(contract_bad.sum())
    n_outside = int(outside.sum())
    n_drift = int(np.any([~exact[l][0] for l in exact], axis=0)[~contract_bad].sum())
    verdict = "FAIL" if n_fail else ("DRIFT-ONLY" if n_drift else "PASS")

    # --- Console report -------------------------------------------------------
    print(f"   {n} exposures, {start.min().isot[:19]} -> {end.max().isot[:19]}, "
          f"median exposure {np.median(dur.value):.0f} s")
    print("   violations per constraint (contract = bracketing grid points | "
          "exact start / middle / end):")
    for j, lab in enumerate(ev.labels):
        c = int((~contract_ops[j]).sum())
        s, m_, e = (int((~exact[l][1][j]).sum()) for l in ("start", "middle", "end"))
        flag = "  <-- VIOLATION" if c else ("  (drift only)" if (s or m_ or e) else "")
        print(f"     {short(lab):<46} {c:>4} | {s:>4} / {m_:>4} / {e:>4}{flag}")
    sfull = [int((~exact[l][0]).sum()) for l in ("start", "middle", "end")]
    print(f"     {'ALL (mission + supplementary)':<46} {int((~contract_full).sum()):>4} | "
          f"{sfull[0]:>4} / {sfull[1]:>4} / {sfull[2]:>4}")
    if n_outside:
        print(f"   {n_outside} exposure(s) outside the scheduler's time grid")
    print(f"   VERDICT: {verdict}"
          + (f" ({n_fail} exposure(s) break the contract)" if n_fail else "")
          + (f" ({n_drift} with continuous-time drift only)" if verdict == "DRIFT-ONLY" else ""))

    # sanity: are the pointings really skygrid fields?
    try:
        sg = mission.skygrid
        sg = sg if isinstance(sg, SkyCoord) else None
        if sg is not None and "field_id" in obs.colnames:
            ref = SkyCoord(sg.ra.unmasked, sg.dec.unmasked) if hasattr(sg.ra, "unmasked") else sg
            sep = coords.separation(ref[np.asarray(obs["field_id"], int)]).arcsec
            print(f"   skygrid check: max |target_coord - skygrid[field_id]| = {sep.max():.3g} arcsec")
    except Exception as e:  # noqa: BLE001
        print(f"   skygrid check skipped ({type(e).__name__}: {e})")

    # --- Violation list -------------------------------------------------------
    if n_fail:
        lst = []
        for i in np.where(contract_bad)[0]:
            bad = [ev.labels[j] for j in range(n_ops) if not contract_ops[j, i]]
            lst.append(dict(row=i, start=start[i].isot[:19], field_id=int(obs["field_id"][i]),
                            ra=coords[i].ra.deg, dec=coords[i].dec.deg,
                            frac_grid_ok=frac_full[i], outside_grid=bool(outside[i]),
                            violated=" ; ".join(bad)))
        df = pd.DataFrame(lst)
        out = path.with_name(f"violations_m4opt_{name}.csv")
        df.to_csv(out, index=False)
        with pd.option_context("display.max_columns", None, "display.width", 250,
                               "display.max_colwidth", 70):
            print(f"   first violating exposures (all in {out.name}):")
            print(df.head(args.max_list).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    res.update(first_obs=start.min().isot[:19], last_obs=end.max().isot[:19],
               time_step_s=ts, n_contract_viol=n_fail, n_outside_grid=n_outside,
               n_drift_only=n_drift, verdict=verdict)

    data = dict(name=name, mission=mission, ev=ev, obs=obs, coords=coords, start=start,
                mid=mid, end=end, frac_ops=frac_ops, frac_full=frac_full,
                contract_bad=contract_bad, contract_ops=contract_ops, exact=exact,
                verdict=verdict, meta=meta, ts=ts, n_fail=n_fail, n_drift=n_drift)
    return res, data


# ----------------------------------------------------------------------------
# Plots (all shading / verdicts come from m4opt constraint evaluations)
# ----------------------------------------------------------------------------
def read_skymap(path):
    from ligo.skymap.io import read_sky_map
    import healpy as hp

    p = np.asarray(read_sky_map(str(path), nest=False)[0], float)
    p = p if p.ndim == 1 else p[0]
    return p, hp.npix2nside(p.size)


def altitude_deg(mission, coords, t):
    """Altitude as m4opt computes it (AltAzConstraint: AltAz frame at the mission's
    observer location)."""
    loc = mission.observer_location(t)
    alt = coords.transform_to(AltAz(obstime=t, location=loc)).alt.deg
    if isinstance(alt, Masked):
        alt = alt.filled(np.nan)
    return np.asarray(alt, float)


def airmass_m4opt(alt_deg):
    """m4opt's airmass model (m4opt.constraints._airmass): airmass = 1 / sin(altitude),
    i.e. sec(zenith angle). Undefined (NaN) below the horizon."""
    alt = np.asarray(alt_deg, float)
    out = np.full(alt.shape, np.nan)
    ok = alt > 0
    out[ok] = 1.0 / np.sin(np.radians(alt[ok]))
    return out


def make_figure(d, path, args):
    ev, mission, coords = d["ev"], d["mission"], d["coords"]
    start, mid, end = d["start"], d["mid"], d["end"]
    n = len(start)
    hours = (start - start.min()).to_value(u.hour)
    bad = d["contract_bad"]
    tref = start[n // 2]
    fig, axs = plt.subplots(1, 3, figsize=(23, 6.6))

    # Sun and Moon at the reference time (display only, astropy ephemerides)
    sun_c = moon_c = None
    try:
        loc_ref = mission.observer_location(tref)
        sun, moon = get_sun(tref), get_body("moon", tref, loc_ref)
        # keep only the direction seen from the observer (GCRS ra/dec), not .icrs
        sun_c = SkyCoord(ra=sun.ra, dec=sun.dec, frame="icrs")
        moon_c = SkyCoord(ra=moon.ra, dec=moon.dec, frame="icrs")
    except Exception as e:  # noqa: BLE001
        print(f"   Sun/Moon not drawn ({type(e).__name__}: {e})")
    moon_sep = next((getattr(op, "min", None) for op in ev.operands
                     if isinstance(op, MoonSeparationConstraint)), None)

    def draw_sun_moon(ax):
        if sun_c is not None:
            ax.plot(sun_c.ra.deg, sun_c.dec.deg, "*", c="gold", mec="k", ms=18,
                    zorder=5, label=f"Sun ({tref.isot[11:16]} UTC)")
        if moon_c is not None:
            ax.plot(moon_c.ra.deg, moon_c.dec.deg, "o", c="orange", mec="k", ms=11,
                    zorder=5, label="Moon")
            if moon_sep is not None:
                r = as_quantity(moon_sep, u.deg)
                ring = moon_c.directional_offset_by(np.linspace(0, 360, 200) * u.deg, r)
                ax.scatter(ring.ra.deg, ring.dec.deg, s=3, c="orange", zorder=4,
                           label=f"Moon separation limit ({r.value:.0f} deg)")

    # (1) sky map + pointings + Sun/Moon
    ax = axs[0]
    skymap_name = meta_get(d["meta"], "skymap")
    for folder in (args.skymap_dir, Path(path).parent):
        cand = Path(folder) / str(skymap_name) if (folder and skymap_name) else None
        if cand is not None and cand.is_file():
            try:
                import healpy as hp
                p, nside = read_skymap(cand)
                RA, DEC = np.meshgrid(np.linspace(0, 360, 361), np.linspace(-90, 90, 181))
                pix = hp.ang2pix(nside, np.radians(90 - DEC), np.radians(RA))
                ax.pcolormesh(RA, DEC, p[pix], cmap="Greys", shading="auto")
            except Exception as e:  # noqa: BLE001
                print(f"   skymap not drawn ({e})")
            break
    sc = ax.scatter(coords.ra.deg, coords.dec.deg, c=hours, s=16, cmap="viridis", zorder=3)
    if bad.any():
        ax.scatter(coords.ra.deg[bad], coords.dec.deg[bad], s=110, facecolors="none",
                   edgecolors="red", lw=1.8, zorder=6, label="breaks m4opt contract")
    gp = SkyCoord(l=np.linspace(0, 360, 1500) * u.deg, b=0 * u.deg, frame="galactic").icrs
    ax.scatter(gp.ra.deg, gp.dec.deg, s=1.2, c="tab:purple", alpha=0.6)
    draw_sun_moon(ax)
    plt.colorbar(sc, ax=ax, label="Hours since first exposure")
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Skymap + pointings (purple: galactic plane)")
    ax.legend(loc="lower left", fontsize=7)

    # (2) m4opt field of regard at the reference time
    ax = axs[1]
    RA, DEC = np.meshgrid(np.arange(0, 360.01, 2.0), np.arange(-90, 90.01, 2.0))
    grid = SkyCoord(RA * u.deg, DEC * u.deg)
    loc = mission.observer_location(tref)
    ok = as_bool(ev.constraint(loc, grid, tref), RA.shape)
    ax.pcolormesh(RA, DEC, ok.astype(float), cmap="Greens", vmin=0, vmax=1.6,
                  shading="auto", alpha=0.55)
    near = np.abs((start - tref).to_value(u.min)) <= args.window_min
    ax.scatter(coords.ra.deg[~near], coords.dec.deg[~near], s=8, c="0.6",
               label="other exposures (other times)")
    ax.scatter(coords.ra.deg[near], coords.dec.deg[near], s=26, c="k", zorder=3,
               label=f"exposures within ±{args.window_min:g} min")
    draw_sun_moon(ax)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"m4opt field of regard at {tref.isot[:16]} UTC (green = allowed)")

    # (3) airmass vs time, with m4opt's model: airmass = 1 / sin(altitude)
    ax = axs[2]
    X = {lab: airmass_m4opt(altitude_deg(mission, coords, t))
         for lab, t in (("start", start), ("middle", mid), ("end", end))}
    h_end = (end - start.min()).to_value(u.hour)
    h_mid = (mid - start.min()).to_value(u.hour)
    lim = None
    for op in ev.operands:
        if isinstance(op, AirmassConstraint):
            lim = 1.0 / np.sin(as_quantity(op.min, u.rad).value)
    ymax = np.nanmax([np.nanmax(X["start"]), np.nanmax(X["end"]), lim or 0]) * 1.08
    if lim is not None:
        ax.axhspan(lim, max(ymax, lim * 1.05), color="red", alpha=0.08)
        ax.axhline(lim, c="r", lw=1.5, label=f"m4opt limit: airmass {lim:.2f}")
    for i in range(n):
        col = "red" if bad[i] else "C0"
        ax.plot([hours[i], h_end[i]], [X["start"][i], X["end"][i]], "-", c=col, lw=1.4)
    ax.scatter(h_mid, X["middle"], s=14, c=["red" if b else "C0" for b in bad], zorder=3,
               label="exposure (line: start -> end, dot: middle)")
    ax.set_ylim(1.0, ymax)
    ax.invert_yaxis()
    ax.set_xlabel("Hours since first exposure")
    ax.set_ylabel("Airmass = 1 / sin(altitude)   (m4opt model)")
    ax.set_title(f"Airmass vs time, max = {np.nanmax(X['middle']):.2f}"
                 + (f" (limit {lim:.2f})" if lim is not None else ""))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{d['name']} ({mission.name}, {n} exposures)  -  VERDICT: {d['verdict']}  "
                 f"[{d['n_fail']} break the m4opt contract]", fontsize=12)
    plt.tight_layout()
    out = Path(path).with_name(f"diag_m4opt_{d['name']}.png")
    plt.savefig(out, dpi=125)
    plt.close(fig)
    print(f"   figure: {out}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--mission", default=None,
                    help="module:attr of a custom Mission (default: name in the file metadata)")
    ap.add_argument("--mission-name", default=None,
                    help="fallback builtin mission name if the metadata has none")
    ap.add_argument("--skymap-dir", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-list", type=int, default=15)
    ap.add_argument("--window-min", type=float, default=20.0,
                    help="half-width (min) of the black-dot window in the field-of-regard panel")
    args = ap.parse_args()

    files = args.files or sorted(map(str, Path(".").glob("schedule_*.ecsv")))
    if not files:
        sys.exit(f"No schedule_*.ecsv found in {Path('.').resolve()}")

    results = []
    for f in files:
        r, d = check_schedule(f, args)
        results.append(r)
        if d is not None and not args.no_plots:
            make_figure(d, f, args)

    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string(index=False))
    df.to_csv("summary_m4opt_constraints.csv", index=False)
    n_fail = int((df["verdict"] == "FAIL").sum())
    print(f"\n{n_fail} schedule(s) break m4opt's own constraints out of {len(df)}.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()