#!/usr/bin/env python
"""Run m4opt multi-mission schedules and plots for a set of overlap weights.

Example
-------
python run_overlap_tests.py ./results /path/to/skymap.fits \
    --missions toy_north toy_south rubin ztf
"""

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

#DEFAULT_WEIGHTS = tuple([Decimal("0.7") + Decimal(index) / 100 for index in range(31)])
#DEFAULT_WEIGHTS = tuple([x for x in DEFAULT_WEIGHTS if x not in [Decimal("0.7"), Decimal("0.75"), Decimal("0.8"), Decimal("0.85"), Decimal("0.9"), Decimal("0.95"), Decimal("1.0")]])
DEFAULT_WEIGHTS = tuple([Decimal(value=index) / 10 for index in range(11)])
#DEFAULT_WEIGHTS = tuple(Decimal('0.50') + Decimal(i) / 100 for i in range(43) if i % 5 != 0)

DEFAULT_MISSIONS = (#"toy_north", 
                    #"toy_south", 
                    "rubin", 
                    "ztf"
                    )
DEFAULT_EVENT_TIME = "2026-04-21T04:00:00"
PLOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "low_proba_issue"
    / "plot_candidates_multi.py"
)


def parse_weights(values: list[str]) -> tuple[Decimal, ...]:
    try:
        weights = tuple(Decimal(value) for value in values)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid overlap weight: {error}") from error
    if not weights:
        raise argparse.ArgumentTypeError("provide at least one overlap weight")
    if any(not weight.is_finite() or not 0 <= weight <= 1 for weight in weights):
        raise argparse.ArgumentTypeError("overlap weights must be between 0 and 1")
    if len(set(weights)) != len(weights):
        raise argparse.ArgumentTypeError("overlap weights must be unique")
    return weights


def _weight_label(weight: Decimal) -> str:
    return f"{weight:.2f}"


def _run(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir.resolve()
    skymap: Path = args.skymap.resolve()
    if not skymap.is_file():
        raise FileNotFoundError(f"Sky map not found: {skymap}")
    if not PLOT_SCRIPT.is_file():
        raise FileNotFoundError(f"Candidate plotting script not found: {PLOT_SCRIPT}")
    if len(args.missions) < 2:
        raise ValueError("At least two missions are required by animate-multi.")
    if len(set(args.missions)) != len(args.missions):
        raise ValueError("Mission names must be unique.")

    m4opt = shutil.which(args.m4opt)
    if m4opt is None:
        raise FileNotFoundError(f"Could not find m4opt executable: {args.m4opt}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [output_dir / f"w_{_weight_label(weight)}" for weight in args.weights]
    existing = [
        path
        for path in run_dirs
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output directories already contain files: {paths}. "
            "Choose a new output directory or pass --overwrite."
        )

    for weight, run_dir in zip(args.weights, run_dirs):
        run_dir.mkdir(parents=True, exist_ok=True)
        schedule_base = run_dir / "schedule"
        candidates_base = run_dir / "candidates"
        command = [
            m4opt,
            "schedule-multi",
            str(skymap),
            str(schedule_base),
            f"--overlap-weight={weight}",
            f"--exptime-min={args.exptime_min}",
            f"--bandpass={args.bandpass}",
            f"--deadline={args.deadline}",
            f"--time-step={args.time_step}",
            f"--max-fields={args.max_fields}",
            f"--timelimit={args.timelimit}",
            f"--jobs={args.jobs}",
            f"--cutoff={0.02}",
            "--write-candidates",
            str(candidates_base),
        ]
        if args.event_time is not None:
            command.append(f"--event-time={args.event_time}")
        command.extend(
            option
            for mission in args.missions
            for option in ("--mission-names", mission)
        )
        print(f"\n=== overlap_weight={weight} ===", flush=True)
        _run(command)

        schedules = [run_dir / f"schedule_{mission}.ecsv" for mission in args.missions]
        candidates = [
            run_dir / f"candidates_{mission}.npz" for mission in args.missions
        ]
        missing = [path for path in (*schedules, *candidates) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "schedule-multi did not create expected outputs: "
                + ", ".join(map(str, missing))
            )

        animation_command = [
            m4opt,
            "animate-multi",
            *(str(path) for path in schedules),
            "--time-step",
            args.animation_time_step,
            "--output",
            str(run_dir / "animate_multi.gif"),
            "--still",
            str(run_dir / "animate_multi.pdf"),
            "--dpi",
            str(args.dpi),
        ]
        _run(animation_command)

        plot_command = [
            sys.executable,
            str(PLOT_SCRIPT),
            "--candidates",
            *(str(path) for path in candidates),
            "--schedule",
            *(str(path) for path in schedules),
            "--labels",
            *(f"{mission} (w={weight:.2f})" for mission in args.missions),
            "--output",
            str(run_dir / f"candidate_fields.png"),
        ]
        _run(plot_command)

    print(f"\nAll runs and plots completed under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for per-weight schedules, plots, and candidate exports",
    )
    parser.add_argument("skymap", type=Path, help="Input multi-order FITS sky map")
    parser.add_argument(
        "--missions",
        nargs="+",
        default=DEFAULT_MISSIONS,
        help=f"Mission names (default: {' '.join(DEFAULT_MISSIONS)})",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=Decimal,
        default=DEFAULT_WEIGHTS,
        metavar="WEIGHT",
        help="Overlap weights from 0 to 1 (default: 0.0 through 1.0 by 0.1)",
    )
    parser.add_argument(
        "--event-time",
        default=DEFAULT_EVENT_TIME,
        help=f"Event time (default: {DEFAULT_EVENT_TIME})",
    )
    parser.add_argument("--exptime-min", default="300s")
    parser.add_argument("--bandpass", default="r")
    parser.add_argument("--deadline", default="6h")
    parser.add_argument("--time-step", default="1min")
    parser.add_argument("--max-fields", type=int, default=50)
    parser.add_argument("--timelimit", default="120s")
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--animation-time-step", default="20min")
    parser.add_argument("--dpi", type=float, default=120)
    parser.add_argument(
        "--m4opt",
        default="m4opt",
        help="m4opt executable or path (default: m4opt on PATH)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse output directories and overwrite files from previous runs",
    )
    args = parser.parse_args()
    try:
        args.weights = parse_weights([str(weight) for weight in args.weights])
        run(args)
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
