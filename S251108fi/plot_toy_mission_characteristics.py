#!/usr/bin/env python3
"""Plot the built-in toy missions without reading an observation schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import ICRS, SkyCoord, SkyOffsetFrame
from matplotlib.patches import Rectangle

from m4opt.missions import toy_north, toy_south


MISSIONS = (toy_north, toy_south)

LAND_DATA = Path(__file__).with_name("data") / "ne_50m_land.geojson"
# Natural Earth 1:50m land polygons; public-domain data from naturalearthdata.com.

MISSION_CONSTRAINTS = {
    "toy_north": (
        "Astronomical night",
        "Airmass ≤ 2",
        "Moon separation ≥ 30°",
    ),
    "toy_south": (
        "Astronomical night",
        "Airmass ≤ 2",
        "Moon separation ≥ 30°",
        "Avoids Sgr A* (10°)",
    ),
}


def _position(mission) -> tuple[float, float, float]:
    location = mission.observer_location.earth_location.to_geodetic()
    return (
        location.lon.to_value(u.deg),
        location.lat.to_value(u.deg),
        location.height.to_value(u.m),
    )


def _plot_world(ax) -> None:
    with LAND_DATA.open(encoding="utf-8") as file:
        land = json.load(file)
    for feature in land["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            outline = np.asarray(polygon[0], dtype=float)
            ax.fill(
                outline[:, 0],
                outline[:, 1],
                facecolor="#e8e4d8",
                edgecolor="#9c998f",
                linewidth=0.35,
                zorder=1,
            )

    colors = ("#2878b5", "#d95f02")
    for mission, color in zip(MISSIONS, colors):
        lon, lat, height = _position(mission)
        ax.scatter(
            lon,
            lat,
            s=95,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
            label=f"{mission.name} ({lat:.2f}°, {lon:.2f}°)",
        )
        ax.annotate(
            mission.name,
            (lon, lat),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=color,
        )

    ax.set(
        xlim=(-180, 180),
        ylim=(-60, 85),
        xlabel="Longitude (deg)",
        ylabel="Latitude (deg)",
        title="Detector locations",
    )
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 86, 30))
    ax.grid(color="#b8b8b8", linewidth=0.5, alpha=0.6)
    ax.text(
        0.01,
        0.015,
        "Natural Earth 1:50m land data",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)


def _plot_skygrid(ax) -> None:
    grids = [mission.skygrid for mission in MISSIONS]
    same_grid = (
        np.array_equal(grids[0].ra.to_value(u.deg), grids[1].ra.to_value(u.deg))
        and np.array_equal(grids[0].dec.to_value(u.deg), grids[1].dec.to_value(u.deg))
    )

    grid_entries = (
        [(grids[0], "#56758a", "toy_north + toy_south (identical grids)")]
        if same_grid
        else [
            (grid, color, f"{mission.name}: {len(grid):,} centers")
            for mission, grid, color in zip(
                MISSIONS, grids, ("#2878b5", "#d95f02")
            )
        ]
    )
    for grid, color, label in grid_entries:
        ra = grid.ra.to_value(u.deg)
        dec = grid.dec.to_value(u.deg)
        longitude = np.deg2rad((ra - 180 + 180) % 360 - 180)
        latitude = np.deg2rad(dec)
        ax.scatter(
            longitude,
            latitude,
            s=4 if same_grid else 7,
            color=color,
            alpha=0.45 if same_grid else 0.68,
            linewidths=0,
            label=label,
            zorder=2,
        )

    npix = len(grids[0])
    nside = int(round(np.sqrt(npix / 12)))
    healpix = HEALPix(nside=nside, frame=ICRS())
    pixel_area = healpix.pixel_area.to_value(u.deg**2)
    ax.set_title("HEALPix sky grid — tile centers", pad=12)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
    sameness = "identical" if same_grid else "different"
    ax.set_xlabel("Right ascension (RA)")
    ax.set_ylabel("Declination")
    ax.text(
        0.5,
        0.98,
        f"Grids {sameness} · nside={nside} · {npix:,} tiles · "
        f"{pixel_area:.2f} deg²/pixel",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _plot_fov_inset(ax, grids[0], same_grid)


def _plot_fov_inset(ax, grid, same_grid: bool) -> None:
    if not same_grid:
        return

    reference = SkyCoord(0 * u.deg, 0 * u.deg, frame=ICRS())
    near_indices = np.flatnonzero(grid.separation(reference).to_value(u.deg) < 8)
    nearby = grid[near_indices]
    separations = nearby[:, np.newaxis].separation(nearby[np.newaxis, :])
    separation_values = separations.to_value(u.deg)
    np.fill_diagonal(separation_values, np.inf)
    first, second = np.unravel_index(
        np.argmin(separation_values), separation_values.shape
    )
    tile_indices = near_indices[[first, second]]
    tiles = grid[tile_indices]

    origin = tiles[0].directional_offset_by(
        tiles[0].position_angle(tiles[1]),
        tiles[0].separation(tiles[1]) / 2,
    )
    offsets = tiles.transform_to(SkyOffsetFrame(origin=origin))
    centers = np.column_stack(
        (offsets.lon.to_value(u.deg), offsets.lat.to_value(u.deg))
    )
    width = MISSIONS[0].fov.width.to_value(u.deg)
    height = MISSIONS[0].fov.height.to_value(u.deg)
    tile_colors = ("#2878b5", "#d95f02")

    inset = ax.inset_axes([0.57, 0.05, 0.40, 0.40], projection="rectilinear")
    bounds = []
    tile_data = zip(
        tile_indices, centers, tile_colors, ((-18, -16), (8, 10))
    )
    for tile_index, (center_x, center_y), tile_color, label_offset in tile_data:
        left, right = center_x - width / 2, center_x + width / 2
        bottom, top = center_y - height / 2, center_y + height / 2
        bounds.append((left, right, bottom, top))
        inset.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor=tile_color,
                edgecolor="#2878b5",
                alpha=0.18,
                linewidth=2,
                zorder=2,
            )
        )
        inset.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor="none",
                edgecolor="#d95f02",
                linestyle=(0, (4, 2)),
                linewidth=1.2,
                zorder=3,
            )
        )
        inset.scatter(center_x, center_y, color="black", s=18, zorder=4)
        inset.annotate(
            f"Tile {tile_index}",
            (center_x, center_y),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=7,
            zorder=5,
        )

    left = max(bounds[0][0], bounds[1][0])
    right = min(bounds[0][1], bounds[1][1])
    bottom = max(bounds[0][2], bounds[1][2])
    top = min(bounds[0][3], bounds[1][3])
    if left < right and bottom < top:
        inset.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                facecolor="#4b3868",
                edgecolor="none",
                alpha=0.24,
                zorder=2.5,
            )
        )
        inset.text(
            (left + right) / 2,
            (bottom + top) / 2,
            "Overlap",
            ha="center",
            va="center",
            fontsize=7,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            zorder=6,
        )

    padding = 0.25
    inset.set_xlim(
        min(bounds[0][0], bounds[1][0]) - padding,
        max(bounds[0][1], bounds[1][1]) + padding,
    )
    inset.set_ylim(
        min(bounds[0][2], bounds[1][2]) - padding,
        max(bounds[0][3], bounds[1][3]) + padding,
    )
    inset.set_aspect("equal", adjustable="box")
    inset.set_title(
        "Adjacent tile FOVs\nNorth solid · South dashed",
        fontsize=8,
        pad=3,
    )
    inset.set_xlabel("Offset longitude (deg)", fontsize=6)
    inset.set_ylabel("Offset latitude (deg)", fontsize=6)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.25, linewidth=0.5)


def _plot_filters(ax) -> None:
    detector = MISSIONS[0].detector
    for band, bandpass in detector.bandpasses.items():
        wavelength = bandpass.waveset.to_value(u.AA)
        transmission = np.asarray(bandpass(bandpass.waveset).value)
        ax.plot(wavelength, transmission, linewidth=2, label=f"{band} filter")

    ax.set(
        xlabel="Wavelength (Å)",
        ylabel="Transmission",
        title="Filter transmission curves (shared by both missions)",
        ylim=(0, None),
    )
    ax.grid(alpha=0.3)
    ax.legend()


def _validate_shared_detector() -> None:
    reference = MISSIONS[0].detector
    scalar_specs = (
        ("area", u.cm**2),
        ("plate_scale", u.arcsec**2),
        ("dark_noise", u.Hz),
    )
    for mission in MISSIONS[1:]:
        detector = mission.detector
        if reference.bandpasses.keys() != detector.bandpasses.keys():
            raise ValueError("The toy missions do not have the same filter names.")
        for attribute, unit in scalar_specs:
            if not np.isclose(
                getattr(reference, attribute).to_value(unit),
                getattr(detector, attribute).to_value(unit),
            ):
                raise ValueError(f"The toy missions have different {attribute} values.")
        for attribute in ("read_noise", "npix", "gain", "aperture_correction"):
            if not np.isclose(
                getattr(reference, attribute), getattr(detector, attribute)
            ):
                raise ValueError(f"The toy missions have different {attribute} values.")
        for band in reference.bandpasses:
            first = reference.bandpasses[band]
            second = detector.bandpasses[band]
            first_waves = first.waveset.to_value(u.AA)
            second_waves = second.waveset.to_value(u.AA)
            if not np.array_equal(first_waves, second_waves) or not np.allclose(
                first(first.waveset).value, second(second.waveset).value
            ):
                raise ValueError(f"The toy missions have different {band} filters.")


def _specification_text() -> str:
    detector = MISSIONS[0].detector
    fov_width = MISSIONS[0].fov.width.to_value(u.deg)
    fov_height = MISSIONS[0].fov.height.to_value(u.deg)
    slew = MISSIONS[0].slew
    plate_scale = detector.plate_scale.to_value(u.arcsec**2)
    pixel_scale = np.sqrt(plate_scale)
    area = detector.area.to_value(u.cm**2)

    lines = [
        "DETECTOR — identical for both missions",
        f"Collecting area: {area:,.0f} cm²",
        f"Pixel scale: {pixel_scale:g}″/pixel ({plate_scale:g} arcsec²/pixel)",
        f"Read noise: {detector.read_noise:g} e⁻",
        f"Dark noise: {detector.dark_noise.to_value(u.Hz):g} s⁻¹",
        f"Effective photometry pixels: {detector.npix:.2f}",
        f"Gain: {detector.gain:g} · aperture correction: {detector.aperture_correction:g}",
        "Background: average sky + zodiacal light",
        f"Field of view: {fov_width:g}° × {fov_height:g}°",
        (
            "Slew: "
            f"{slew.max_angular_velocity.to_value(u.deg / u.s):g}°/s, "
            f"{slew.max_angular_acceleration.to_value(u.deg / u.s**2):g}°/s², "
            f"pose {slew.settling_time.to_value(u.s):g} s"
        ),
        "",
    ]

    for mission in MISSIONS:
        lon, lat, height = _position(mission)
        lines.extend(
            [
                f"{mission.name} — {lat:.2f}° lat, {lon:.2f}° lon, {height:.0f} m",
                "Constraints: " + " ; ".join(MISSION_CONSTRAINTS[mission.name]),
            ]
        )
    return "\n".join(lines)


def make_plot(output: Path, show: bool = False) -> None:
    _validate_shared_detector()
    fig = plt.figure(figsize=(19, 12), layout="constrained")
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1, 1.45),
        height_ratios=(1, 0.9, 1.1),
    )
    world_ax = fig.add_subplot(grid[0, 0])
    filters_ax = fig.add_subplot(grid[1, 0])
    specs_ax = fig.add_subplot(grid[2, 0])
    skygrid_ax = fig.add_subplot(grid[:, 1], projection="mollweide")
    _plot_world(world_ax)
    _plot_skygrid(skygrid_ax)
    _plot_filters(filters_ax)

    specs_ax.axis("off")
    specs_ax.set_title("Mission characteristics")
    specs_ax.text(
        0.02,
        0.98,
        _specification_text(),
        transform=specs_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.5,
        family="monospace",
    )

    fig.suptitle("Toy missions — sites, sky grid, and detector", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Plot written to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("toy_mission_characteristics.png"),
        help="Output image path (PNG).",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot.")
    args = parser.parse_args()
    make_plot(args.output, args.show)


if __name__ == "__main__":
    main()
