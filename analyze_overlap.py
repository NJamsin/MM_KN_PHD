#!/usr/bin/env python
"""Summarize overlap-weight schedule runs and create comparison figures."""

import argparse
import csv
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.path import Path as MatplotlibPath
from scipy.spatial import ConvexHull
from astropy.coordinates import ICRS
from astropy.table import QTable
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map


WEIGHTS = np.arange(0.0,1.05,0.05)
WEIGHTS2 = np.array([0.7 + i/100 for i in range(31)])
WEIGHTS = np.unique(np.concatenate((WEIGHTS, WEIGHTS2)))
MISSIONS = ("toy_north", 
            "toy_south", 
            #"rubin", 
            #"ztf"
)
MISSION_COLORS = {
    "toy_north": "tab:blue",
    "toy_south": "tab:orange",
    #"rubin": "tab:green",
    #"ztf": "tab:red",
}


def _read_run(run_dir, missions, pixel_probability, nside):
    row = {"overlap_weight": float(run_dir.name.removeprefix("w_"))}
    scheduled_pixels = []
    candidate_pixels = []
    candidate_centers = []
    scheduled_pixels_by_mission = {}
    gross_probability = 0.0
    mission_field_counts = {}
    mission_centers = {}
    statuses = []
    solution_times = []

    for mission in missions:
        candidates = np.load(run_dir / f"candidates_{mission}.npz")
        schedule = QTable.read(
            run_dir / f"schedule_{mission}.ecsv", format="ascii.ecsv"
        )
        field_ids = np.asarray(candidates["field_id"])
        offsets = candidates["footprint_offsets"]
        pixels = candidates["footprint_pixels"]
        footprints = [
            pixels[offsets[index] : offsets[index + 1]]
            for index in range(len(field_ids))
        ]
        field_indices = {int(field_id): index for index, field_id in enumerate(field_ids)}

        observations = schedule[schedule["action"] == "observe"]
        scheduled_ids = np.unique(np.asarray(observations["field_id"], dtype=int))
        missing_ids = [int(field_id) for field_id in scheduled_ids if int(field_id) not in field_indices]
        if missing_ids:
            raise ValueError(f"{mission} schedule fields missing from candidates: {missing_ids}")

        selected_indices = [field_indices[int(field_id)] for field_id in scheduled_ids]
        mission_scheduled_pixels = [
            footprints[index] for index in selected_indices
        ]
        scheduled_pixels.extend(mission_scheduled_pixels)
        scheduled_pixels_by_mission[mission] = (
            np.unique(np.concatenate(mission_scheduled_pixels))
            if mission_scheduled_pixels
            else np.asarray([], dtype=np.intp)
        )
        candidate_pixels.extend(footprints)
        candidate_centers.append(
            np.column_stack(
                (
                    np.asarray(candidates["ra_deg"]),
                    np.asarray(candidates["dec_deg"]),
                )
            )
        )
        gross_probability += float(np.asarray(candidates["probability"])[selected_indices].sum())
        mission_field_counts[mission] = len(selected_indices)
        mission_centers[mission] = (
            np.asarray(candidates["ra_deg"])[selected_indices],
            np.asarray(candidates["dec_deg"])[selected_indices],
        )
        statuses.append(str(schedule.meta.get("solution_status", "unknown")))
        solution_time = schedule.meta.get("solution_time")
        solution_times.append(
            float(solution_time.to_value("s"))
            if hasattr(solution_time, "to_value")
            else float(solution_time or 0)
        )

    scheduled_union = (
        np.unique(np.concatenate(scheduled_pixels))
        if scheduled_pixels
        else np.asarray([], dtype=np.intp)
    )
    candidate_union = (
        np.unique(np.concatenate(candidate_pixels))
        if candidate_pixels
        else np.asarray([], dtype=np.intp)
    )
    candidate_envelope = _candidate_envelope(
        np.concatenate(candidate_centers),
        nside,
    )
    overlap_pixels = (
        scheduled_pixels_by_mission[missions[0]]
        if missions
        else np.asarray([], dtype=np.intp)
    )
    for mission in missions[1:]:
        overlap_pixels = np.intersect1d(
            overlap_pixels,
            scheduled_pixels_by_mission[mission],
            assume_unique=True,
        )

    gap_candidates = (
        np.unique(
            hp.get_all_neighbours(nside, scheduled_union, nest=True)
        )
        if scheduled_union.size
        else np.asarray([], dtype=np.intp)
    )
    gap_candidates = gap_candidates[gap_candidates >= 0]
    gap_candidates = gap_candidates[
        ~np.isin(gap_candidates, scheduled_union)
    ]
    gap_neighbours = (
        hp.get_all_neighbours(
            nside,
            gap_candidates,
            nest=True,
        )
        if gap_candidates.size
        else np.empty((8, 0), dtype=np.intp)
    )
    scheduled_neighbour_counts = np.sum(
        (gap_neighbours >= 0)
        & np.isin(gap_neighbours, scheduled_union),
        axis=0,
    )
    strict_gap_pixels = gap_candidates[scheduled_neighbour_counts >= 6]
    gap_pixels = np.setdiff1d(
        candidate_envelope,
        scheduled_union,
        assume_unique=True,
    )
    candidate_gap_pixels = np.intersect1d(
        gap_pixels,
        candidate_union,
        assume_unique=True,
    )
    interstitial_gap_pixels = np.setdiff1d(
        gap_pixels,
        candidate_union,
        assume_unique=True,
    )
    strict_gap_inside_envelope = np.intersect1d(
        strict_gap_pixels,
        gap_pixels,
        assume_unique=True,
    )
    row.update(
        scheduled_unique_probability=float(pixel_probability[scheduled_union].sum()),
        candidate_union_probability=float(pixel_probability[candidate_union].sum()),
        overlap_intersection_probability=float(pixel_probability[overlap_pixels].sum()),
        gap_probability=float(pixel_probability[gap_pixels].sum()),
        strict_gap_probability=float(pixel_probability[strict_gap_pixels].sum()),
        gross_field_probability=gross_probability,
        overlap_excess_probability=gross_probability
        - float(pixel_probability[scheduled_union].sum()),
        scheduled_unique_pixels=len(scheduled_union),
        candidate_unique_pixels=len(candidate_union),
        overlap_intersection_pixels=len(overlap_pixels),
        gap_pixels=len(gap_pixels),
        gap_candidate_pixels=len(candidate_gap_pixels),
        gap_interstitial_pixels=len(interstitial_gap_pixels),
        gap_envelope_pixels=len(candidate_envelope),
        strict_gap_pixels=len(strict_gap_pixels),
        strict_gap_inside_envelope_pixels=len(strict_gap_inside_envelope),
        scheduled_pixel_indices=scheduled_union,
        candidate_envelope_pixel_indices=candidate_envelope,
        gap_pixel_indices=gap_pixels,
        candidate_gap_pixel_indices=candidate_gap_pixels,
        interstitial_gap_pixel_indices=interstitial_gap_pixels,
        strict_gap_pixel_indices=strict_gap_inside_envelope,
        solver_status="; ".join(dict.fromkeys(statuses)),
        solution_time_s=max(solution_times, default=0.0),
        mission_field_counts=mission_field_counts,
        mission_centers=mission_centers,
    )
    return row


def _candidate_envelope(candidate_centers_deg, nside):
    # The candidate-center hull limits gap counting to the interior survey region.
    ra, dec = np.deg2rad(candidate_centers_deg).T
    vectors = np.column_stack(
        (
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        )
    )
    center = vectors.mean(axis=0)
    center /= np.linalg.norm(center)

    east = np.array((-center[1], center[0], 0.0))
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    center_distance = vectors @ center
    if np.any(center_distance <= 0):
        raise ValueError("Candidate centers span more than one tangent hemisphere")

    projected = np.column_stack(
        (
            vectors @ east / center_distance,
            vectors @ north / center_distance,
        )
    )
    hull = ConvexHull(projected)
    radius = np.max(np.arccos(np.clip(center_distance, -1.0, 1.0)))
    radius += 2.0 / nside
    envelope_pixels = hp.query_disc(
        nside,
        center,
        radius,
        inclusive=True,
        nest=True,
        fact=4,
    )
    envelope_vectors = np.asarray(
        hp.pix2vec(nside, envelope_pixels, nest=True)
    ).T
    envelope_distance = envelope_vectors @ center
    if np.any(envelope_distance <= 0):
        raise ValueError("Candidate-center envelope spans more than one tangent hemisphere")
    envelope_projected = np.column_stack(
        (
            envelope_vectors @ east / envelope_distance,
            envelope_vectors @ north / envelope_distance,
        )
    )
    inside_hull = MatplotlibPath(projected[hull.vertices]).contains_points(
        envelope_projected,
        radius=1e-12,
    )
    return envelope_pixels[inside_hull]


def _write_gap_map(row, nside, output):
    sky = np.full(hp.nside2npix(nside), hp.UNSEEN)
    sky[row["scheduled_pixel_indices"]] = 0
    sky[row["interstitial_gap_pixel_indices"]] = 1
    sky[row["candidate_gap_pixel_indices"]] = 2

    cmap = ListedColormap(("lightgray", "crimson", "orange"))
    center_vector = np.asarray(
        hp.pix2vec(
            nside,
            row["candidate_envelope_pixel_indices"],
            nest=True,
        ),
    ).mean(axis=1)
    center_theta, center_phi = hp.vec2ang(center_vector)
    center_theta = float(np.asarray(center_theta).item())
    center_phi = float(np.asarray(center_phi).item())
    hp.gnomview(
        sky,
        rot=(
            np.degrees(center_phi),
            90 - np.degrees(center_theta),
            0,
        ),
        nest=True,
        title=(
            f"Candidate-center-envelope gaps (overlap_weight="
            f"{row['overlap_weight']:.2f})"
        ),
        xsize=1600,
        ysize=1200,
        reso=2,
        min=0,
        max=2,
        cmap=cmap,
        cbar=False,
        badcolor="white",
        bgcolor="white",
    )
    if row["strict_gap_pixel_indices"].size:
        strict_theta, strict_phi = hp.pix2ang(
            nside,
            row["strict_gap_pixel_indices"],
            nest=True,
        )
        hp.projplot(
            strict_theta,
            strict_phi,
            "o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
    fig = plt.gcf()
    fig.legend(
        handles=(
            Patch(
                facecolor="lightgray",
                label=f"Planned ({len(row['scheduled_pixel_indices'])} px)",
            ),
            Patch(
                facecolor="orange",
                label=(
                    "Unplanned candidate pixels "
                    f"({row['gap_candidate_pixels']} px)"
                ),
            ),
            Patch(
                facecolor="crimson",
                label=(
                    "Interstitial pixels inside candidate-center hull "
                    f"({row['gap_interstitial_pixels']} px)"
                ),
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="none",
                label=(
                    "Old >=6/8 criterion inside hull "
                    f"({row['strict_gap_inside_envelope_pixels']} px)"
                ),
            ),
        ),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_gap_probability_figure(rows, output):
    weights = [row["overlap_weight"] for row in rows]
    probabilities = [100 * row["gap_probability"] for row in rows]
    strict_probabilities = [
        100 * row["strict_gap_probability"] for row in rows
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        weights,
        probabilities,
        "d-",
        color="crimson",
        label="Candidate-center-envelope gaps",
    )
    ax.plot(
        weights,
        strict_probabilities,
        "o:",
        color="black",
        label="Old >=6/8-neighbour criterion",
    )
    ax.set_xlabel("overlap_weight")
    ax.set_ylabel("Gap probability (%)")
    ax.set_title("Gap probability")
    ax.set_xticks(weights[::2])
    ax.set_xlim(weights[0] - 0.02, weights[-1] + 0.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_metrics_figure(rows, output):
    weights = [row["overlap_weight"] for row in rows]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))

    ax.plot(
        weights,
        [100 * row["scheduled_unique_probability"] for row in rows],
        "o-",
        label="Planned-field union",
    )
    ax.plot(
        weights,
        [100 * row["candidate_union_probability"] for row in rows],
        "s--",
        label="Preplanned candidate union",
    )
    ax.plot(
        weights,
        [100 * row["overlap_intersection_probability"] for row in rows],
        "^-.",
        label="Shared coverage (both telescopes)",
    )
    ax.plot(
        weights,
        [100 * row["gap_probability"] for row in rows],
        "d:",
        label="Gaps inside candidate-center envelope",
    )
    ax.plot(
        weights,
        [100 * row["strict_gap_probability"] for row in rows],
        "x:",
        label="Strict gaps (old >=6/8 criterion)",
    )
    ax.set_xlabel("overlap_weight")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Probability by overlap_weight")
    ax.set_xticks(weights)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--skymap",
        type=Path,
        default=Path("/home/liteul/phd/events/skymaps/fake_gaussian_30.fits"),
    )
    args = parser.parse_args()

    runs = [
        args.root / f"w_{weight:.2f}"
        for weight in WEIGHTS
    ]
    candidates = np.load(runs[0] / f"candidates_{MISSIONS[0]}.npz")
    nside = int(candidates["nside"])
    sky_map = read_sky_map(args.skymap, moc=True)
    probability = np.asarray(rasterize(sky_map, int(np.log2(nside)))["PROB"])
    rows = [_read_run(run_dir, MISSIONS, probability, nside) for run_dir in runs]

    metrics_path = args.root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "overlap_weight",
                "candidate_union_probability",
                "scheduled_unique_probability",
                "overlap_intersection_probability",
                "gap_probability",
                "strict_gap_probability",
                "gross_field_probability",
                "overlap_excess_probability",
                "scheduled_unique_pixels",
                "candidate_unique_pixels",
                "overlap_intersection_pixels",
                "gap_pixels",
                "gap_candidate_pixels",
                "gap_interstitial_pixels",
                "gap_envelope_pixels",
                "strict_gap_pixels",
                "strict_gap_inside_envelope_pixels",
                "solver_status",
                "solution_time_s",
                *(f"{mission}_fields" for mission in MISSIONS),
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value
                    for key, value in {
                        **row,
                        **{
                            f"{mission}_fields": row["mission_field_counts"][mission]
                            for mission in MISSIONS
                        },
                    }.items()
                    if key in writer.fieldnames
                }
            )

    figure_metrics = args.root / "overlap_metrics.png"
    figure_gaps = args.root / "gap_probability.png"
    figure_sky = args.root / "coverage_fields_w0_vs_w1.png"
    report_path = args.root / "REPORT.md"
    _write_metrics_figure(rows, figure_metrics)
    _write_gap_probability_figure(rows, figure_gaps)
    gap_maps_dir = args.root / "gap_maps"
    gap_maps_dir.mkdir(exist_ok=True)
    for row in rows:
        gap_map_path = gap_maps_dir / f"gaps_w_{row['overlap_weight']:.2f}.png"
        _write_gap_map(row, nside, gap_map_path)
    print(
        f"Wrote {metrics_path}, {figure_metrics}, {figure_sky}, "
        f"{figure_gaps}, {report_path}, and gap maps in {gap_maps_dir}"
    )


if __name__ == "__main__":
    main()
