#!/usr/bin/env python
"""Compare multiple multi-order FITS sky maps on one color-coded sky plot.

By default, all ``T*.fits`` files beside this script are plotted. The 50% and
90% credible-region contours use solid and dashed lines, respectively.

Example
-------
python plot_skymaps.py
python plot_skymaps.py T0_control.fits T1_airmass_dec-40.fits --output compare.png
"""

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map


def credible_density_thresholds(probability: np.ndarray) -> tuple[float, float]:
    """Return the probability-density thresholds enclosing 50% and 90%."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or probability.size == 0:
        raise ValueError("Expected a non-empty one-dimensional sky map.")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError("Sky-map probabilities must be finite and non-negative.")
    total = probability.sum()
    if total <= 0:
        raise ValueError("Sky-map probability sum must be positive.")

    ordered = np.sort(probability.ravel())[::-1]
    cumulative = np.cumsum(ordered) / total
    thresholds = [
        float(ordered[min(np.searchsorted(cumulative, level), len(ordered) - 1)])
        for level in (0.5, 0.9)
    ]
    return thresholds[0], thresholds[1]


def plot_skymaps(paths: list[Path], output: Path, nside: int) -> None:
    if not paths:
        raise ValueError("No sky maps were supplied.")
    if nside < 32 or nside & (nside - 1):
        raise ValueError("nside must be a power of two greater than or equal to 32.")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Sky map(s) not found: " + ", ".join(map(str, missing)))

    projector = hp.projector.MollweideProj(xsize=1200)
    xmin, xmax, ymin, ymax = projector.get_extent()
    x_grid, y_grid = projector.ij2xy()

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = plt.get_cmap("tab10").colors
    legend_handles = []

    for index, path in enumerate(paths):
        color = colors[index % len(colors)]
        sky_map = read_sky_map(path, moc=True)
        flat_map = rasterize(sky_map, int(np.log2(nside)))
        probability = np.asarray(flat_map["PROB"], dtype=float)
        threshold_50, threshold_90 = credible_density_thresholds(probability)
        projected_probability = np.ma.masked_invalid(
            projector.projmap(
                probability,
                lambda x, y, z: hp.vec2pix(
                    nside,
                    x,
                    y,
                    z,
                    nest=True,
                ),
            )
        )
        ax.contour(
            x_grid,
            y_grid,
            projected_probability,
            levels=[threshold_90],
            colors=[color],
            linewidths=1.5,
            linestyles="--",
        )
        ax.contour(
            x_grid,
            y_grid,
            projected_probability,
            levels=[threshold_50],
            colors=[color],
            linewidths=2.2,
            linestyles="-",
        )
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color=color,
                linewidth=2.2,
                label=path.stem,
            )
        )

    for ra in np.arange(0, 360, 30):
        dec_line = np.linspace(-89.9, 89.9, 500)
        grid_x, grid_y = projector.ang2xy(
            np.full_like(dec_line, ra), dec_line, lonlat=True
        )
        ax.plot(grid_x, grid_y, color="0.82", linewidth=0.6, zorder=0)

    for dec in np.arange(-75, 90, 15):
        ra_line = np.linspace(0, 360, 721)
        grid_x, grid_y = projector.ang2xy(
            ra_line, np.full_like(ra_line, dec), lonlat=True
        )
        ax.plot(grid_x, grid_y, color="0.82", linewidth=0.6, zorder=0)

    boundary_angle = np.linspace(0, 2 * np.pi, 721)
    ax.plot(
        2 * np.cos(boundary_angle),
        np.sin(boundary_angle),
        color="0.25",
        linewidth=1,
        zorder=0,
    )

    ax.legend(
        handles=[
            plt.Line2D([], [], color="0.2", linewidth=2.2, linestyle="-", label="50%"),
            plt.Line2D([], [], color="0.2", linewidth=1.5, linestyle="--", label="90%"),
        ],
        title="Région de crédibilité",
        loc="upper right",
    )
    ra_ticks = np.arange(0, 360, 30)
    ra_tick_x, _ = projector.ang2xy(ra_ticks, np.zeros_like(ra_ticks), lonlat=True)
    ra_tick_x = np.append(ra_tick_x, xmin)
    ra_labels = [f"{ra}°" for ra in ra_ticks] + ["180°"]
    ra_order = np.argsort(ra_tick_x)
    dec_ticks = np.arange(-75, 90, 15)
    _, dec_tick_y = projector.ang2xy(
        np.full_like(dec_ticks, 180), dec_ticks, lonlat=True
    )
    ax.set_xticks(ra_tick_x[ra_order])
    ax.set_xticklabels([ra_labels[index] for index in ra_order])
    ax.set_yticks(dec_tick_y)
    ax.set_yticklabels([f"{dec}°" for dec in dec_ticks])
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    fig.legend(
        handles=legend_handles,
        title="Skymaps",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(paths)),
        frameon=True,
    )
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted {len(paths)} sky maps to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "skymaps",
        nargs="*",
        type=Path,
        help="Input FITS sky maps (default: T*.fits in the script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("skymap_comparison.png"),
        help="Output plot path (default: skymap_comparison.png)",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="Rasterization HEALPix nside (default: 128)",
    )
    args = parser.parse_args()

    paths = args.skymaps or sorted(Path(__file__).parent.glob("T*.fits"))
    try:
        plot_skymaps(paths, args.output, args.nside)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
