#!/usr/bin/env python
"""Plot candidate Rubin fields and the fields selected by a schedule."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import ICRS
from astropy.table import QTable
from astropy_healpix import HEALPix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(__file__).with_name("candidates_rubin_100fields_test.npz"),
        help="Candidate footprint export produced by --write-candidates",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(__file__).with_name(
            "schedule_rubin_100fields_candidates_test.ecsv"
        ),
        help="Schedule ECSV produced by m4opt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("candidate_fields.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    candidates = np.load(args.candidates)
    schedule = QTable.read(args.schedule, format="ascii.ecsv")

    field_ids = candidates["field_id"]
    offsets = candidates["footprint_offsets"]
    pixels = candidates["footprint_pixels"]
    if len(offsets) != len(field_ids) + 1 or offsets[-1] != len(pixels):
        raise ValueError("Candidate footprint offsets do not match the candidate data")

    observations = schedule[schedule["action"] == "observe"]
    scheduled_ids = np.unique(observations["field_id"])
    scheduled_mask = np.isin(field_ids, scheduled_ids)
    missing_ids = np.setdiff1d(scheduled_ids, field_ids)
    if len(missing_ids):
        raise ValueError(
            "Schedule contains fields missing from the candidate export: "
            f"{missing_ids.tolist()}"
        )

    all_footprints = [
        pixels[offsets[index] : offsets[index + 1]]
        for index in range(len(field_ids))
    ]
    candidate_pixels = np.unique(np.concatenate(all_footprints))
    scheduled_footprints = [
        footprint
        for footprint, is_scheduled in zip(all_footprints, scheduled_mask)
        if is_scheduled
    ]
    scheduled_pixels = (
        np.unique(np.concatenate(scheduled_footprints))
        if scheduled_footprints
        else np.asarray([], dtype=np.intp)
    )
    unselected_pixels = np.setdiff1d(candidate_pixels, scheduled_pixels)

    hpx = HEALPix(
        nside=int(candidates["nside"]),
        frame=ICRS(),
        order=str(candidates["order"]),
    )
    candidate_coords = hpx.healpix_to_skycoord(candidate_pixels)
    unselected_coords = hpx.healpix_to_skycoord(unselected_pixels)
    scheduled_coords = hpx.healpix_to_skycoord(scheduled_pixels)

    def mollweide_x(ra_deg):
        return np.deg2rad((180 - ra_deg + 180) % 360 - 180)

    candidate_x = mollweide_x(candidates["ra_deg"])
    candidate_y = np.deg2rad(candidates["dec_deg"])
    ra_hours = np.arange(0, 24, 2)
    tick_x = np.deg2rad((180 - ra_hours * 15 + 180) % 360 - 180)

    ra_center = np.arctan2(
        np.sin(np.deg2rad(candidates["ra_deg"])).mean(),
        np.cos(np.deg2rad(candidates["ra_deg"])).mean(),
    )

    def local_x(ra_deg):
        return np.rad2deg(
            (np.deg2rad(ra_deg) - ra_center + np.pi) % (2 * np.pi) - np.pi
        )

    def add_footprints(ax, x_transform, y_transform):
        ax.scatter(
            x_transform(unselected_coords.ra.deg),
            y_transform(unselected_coords.dec.deg),
            s=1.2,
            color="cornflowerblue",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            label="Candidate footprints (not scheduled)",
        )
        ax.scatter(
            x_transform(scheduled_coords.ra.deg),
            y_transform(scheduled_coords.dec.deg),
            s=1.2,
            color="crimson",
            alpha=0.8,
            linewidths=0,
            rasterized=True,
            label="Scheduled footprints",
        )
        ax.scatter(
            x_transform(candidates["ra_deg"][~scheduled_mask]),
            y_transform(candidates["dec_deg"][~scheduled_mask]),
            marker="x",
            s=13,
            color="navy",
            linewidths=0.7,
            label="Candidate field centers",
        )
        ax.scatter(
            x_transform(candidates["ra_deg"][scheduled_mask]),
            y_transform(candidates["dec_deg"][scheduled_mask]),
            marker="o",
            s=20,
            facecolors="none",
            edgecolors="darkred",
            linewidths=0.9,
            label="Scheduled field centers",
        )

    fig = plt.figure(figsize=(15, 12))
    ax_sky = fig.add_subplot(2, 2, 1, projection="mollweide")
    add_footprints(ax_sky, mollweide_x, np.deg2rad)
    ax_sky.set_xticks(tick_x)
    ax_sky.set_xticklabels([f"{hour:02d}h" for hour in ra_hours])
    ax_sky.grid(True, alpha=0.35)
    ax_sky.set_title("All-sky view")

    ax_zoom = fig.add_subplot(2, 2, 2)
    add_footprints(ax_zoom, local_x, lambda dec_deg: dec_deg)
    x_values = local_x(unselected_coords.ra.deg)
    y_values = unselected_coords.dec.deg
    if len(scheduled_pixels):
        x_values = np.concatenate((x_values, local_x(scheduled_coords.ra.deg)))
        y_values = np.concatenate((y_values, scheduled_coords.dec.deg))
    x_margin = max(np.ptp(x_values) * 0.08, 0.1)
    y_margin = max(np.ptp(y_values) * 0.08, 0.1)
    ax_zoom.set_xlim(x_values.max() + x_margin, x_values.min() - x_margin)
    ax_zoom.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.set_xlabel(r"$\Delta$RA from candidate-field center (deg)")
    ax_zoom.set_ylabel("Declination (deg)")
    ax_zoom.grid(True, alpha=0.35)
    ax_zoom.set_title("Footprint detail (RA increases to the left)")

    ax_candidates = fig.add_subplot(2, 2, 3)
    ax_candidates.scatter(
        local_x(candidate_coords.ra.deg),
        candidate_coords.dec.deg,
        s=1.2,
        color="cornflowerblue",
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        label="Candidate footprints",
    )
    ax_candidates.scatter(
        local_x(candidates["ra_deg"]),
        candidates["dec_deg"],
        marker="x",
        s=13,
        color="navy",
        linewidths=0.7,
        label="Candidate field centers",
    )
    ax_candidates.set_xlim(ax_zoom.get_xlim())
    ax_candidates.set_ylim(ax_zoom.get_ylim())
    ax_candidates.set_aspect("equal", adjustable="box")
    ax_candidates.set_xlabel(r"$\Delta$RA from candidate-field center (deg)")
    ax_candidates.set_ylabel("Declination (deg)")
    ax_candidates.grid(True, alpha=0.35)
    ax_candidates.set_title(f"All {len(field_ids)} candidate fields")
    ax_candidates.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    ax_scheduled = fig.add_subplot(2, 2, 4)
    ax_scheduled.scatter(
        local_x(scheduled_coords.ra.deg),
        scheduled_coords.dec.deg,
        s=1.2,
        color="crimson",
        alpha=0.8,
        linewidths=0,
        rasterized=True,
        label="Scheduled footprints",
    )
    ax_scheduled.scatter(
        local_x(candidates["ra_deg"][scheduled_mask]),
        candidates["dec_deg"][scheduled_mask],
        marker="o",
        s=20,
        facecolors="none",
        edgecolors="darkred",
        linewidths=0.9,
        label="Scheduled field centers",
    )
    ax_scheduled.set_xlim(ax_zoom.get_xlim())
    ax_scheduled.set_ylim(ax_zoom.get_ylim())
    ax_scheduled.set_aspect("equal", adjustable="box")
    ax_scheduled.set_xlabel(r"$\Delta$RA from candidate-field center (deg)")
    ax_scheduled.set_ylabel("Declination (deg)")
    ax_scheduled.grid(True, alpha=0.35)
    ax_scheduled.set_title(
        f"Scheduled fields ({np.count_nonzero(scheduled_mask)})"
    )
    ax_scheduled.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    fig.suptitle(
        f"Observable candidates: {len(field_ids)} | "
        f"Scheduled fields: {np.count_nonzero(scheduled_mask)}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    print(f"Candidate fields: {len(field_ids)}")
    print(f"Scheduled fields: {np.count_nonzero(scheduled_mask)}")
    print(f"Scheduled observations (including repeated visits): {len(observations)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
