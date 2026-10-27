from nmma.em.model import FiestaKilonovaModel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from peket.kn_side.utils import abs_to_app_mag

OUTDIR = "/home/liteul/phd/NSBH/KN_range/H4_chi07"
PLOT_DIR = f"{OUTDIR}/plot"

MAG_LIMITS = {
    "best": {"lsstg": 24.8, "lsstr": 24.4, "lssti": 23.9},
    "worst": {"lsstg": 24.2, "lsstr": 23.7, "lssti": 23.4},
}
ENVELOPE_PERCENTILES = [5, 50, 95]

fiesta_kn_name = "Bu2019nsbh_MLP"

filters = ["lsstg", "lsstr", "lssti"]

fiesta_kn_model = FiestaKilonovaModel(model=fiesta_kn_name, filters=filters)

sample_times = np.arange(0.5,12.5,0.5) 

seed_H_file = f"{OUTDIR}/seed.csv"

seed = pd.read_csv(seed_H_file, sep=',')

params_name = [
    "log10_mej_dyn",
    "log10_mej_wind", 
]

DLs = [40,150,300,400]

GR = {}
RI = {}
GI = {}

for incl, theta in zip(["on", "off"], [0,90]):
    print(f"Incl {incl}")
    GR[incl] = {dl: [] for dl in DLs}
    RI[incl] = {dl: [] for dl in DLs}
    GI[incl] = {dl: [] for dl in DLs}
    for dl in DLs:
        print(f"Dist {dl}")
        GR[incl][dl] = {scenario: [] for scenario in MAG_LIMITS}
        RI[incl][dl] = {scenario: [] for scenario in MAG_LIMITS}
        GI[incl][dl] = {scenario: [] for scenario in MAG_LIMITS}
        for i in tqdm(range(len(seed['redshift']))):
            parameters = {
                name: seed[name][i] for name in params_name
            }
            parameters['KNtheta'] = theta
            parameters['luminosity_distance'] = dl

            mag = fiesta_kn_model.generate_lightcurve(sample_times, parameters)
            mag = abs_to_app_mag(mag, dl)
            for scenario, limits in MAG_LIMITS.items():
                detected_gr = (
                    (mag["lsstg"] <= limits["lsstg"])
                    & (mag["lsstr"] <= limits["lsstr"])
                )
                detected_ri = (
                    (mag["lsstr"] <= limits["lsstr"])
                    & (mag["lssti"] <= limits["lssti"])
                )
                detected_gi = (
                                    (mag["lsstg"] <= limits["lsstg"])
                                    & (mag["lssti"] <= limits["lssti"])
                                )
                
                GR[incl][dl][scenario].append(
                    np.where(detected_gr, mag["lsstg"] - mag["lsstr"], np.nan)
                )
                RI[incl][dl][scenario].append(
                    np.where(detected_ri, mag["lsstr"] - mag["lssti"], np.nan)
                )
                GI[incl][dl][scenario].append(
                                    np.where(detected_gi, mag["lsstg"] - mag["lssti"], np.nan)
                                )

TIME_BIN_WIDTH = 0.5
INCLINATIONS = [("on", "Face-on"), ("off", "Edge-on")]
COMPARISONS = {
    "best": ["best"],
    "worst": ["worst"],
    "best_worst": ["best", "worst"],
}


def summarize_curves(curves):
    """Return the 5th, 50th, and 95th percentiles and number of contributors."""
    values = np.stack(curves)
    summary = np.full((len(ENVELOPE_PERCENTILES), values.shape[1]), np.nan)
    counts = np.isfinite(values).sum(axis=0)

    for time_index in range(values.shape[1]):
        detected_values = values[:, time_index]
        detected_values = detected_values[np.isfinite(detected_values)]
        if detected_values.size:
            summary[:, time_index] = np.percentile(
                detected_values, ENVELOPE_PERCENTILES
            )

    return summary, counts, values.shape[0]


def plot_comparison(color_data, color_label, filename_prefix, comparison, dls, suffix):
    """Save one comparison for the two inclinations and selected distances."""
    fig, axes = plt.subplots(figsize=(16, 6), ncols=2, sharex=True, sharey=True)
    colorbars = {}
    max_counts = {}
    for scenario in COMPARISONS[comparison]:
        max_counts[scenario] = max(
            np.isfinite(
                np.stack(
                    [
                        curve
                        for dl in dls
                        for curve in color_data[incl][dl][scenario]
                    ]
                )
            )
            .sum(axis=0)
            .max()
            for incl, _ in INCLINATIONS
        )

    for ax, (incl, title) in zip(axes, INCLINATIONS):
        for scenario in COMPARISONS[comparison]:
            curves = [
                curve
                for dl in dls
                for curve in color_data[incl][dl][scenario]
            ]
            (lower, median, upper), counts, _ = summarize_curves(curves)
            cmap = (
                mpl.colormaps["seismic"]
                if scenario == "best"
                else mpl.colormaps["seismic_r"]
            )
            half_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f"{cmap.name}_first_half", cmap(np.linspace(0.5, 0.95, 256))
            )
            color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
            colorbars[scenario] = mpl.cm.ScalarMappable(
                norm=color_norm, cmap=half_cmap
            )
            color = cmap(0)
            relative_counts = (
                counts / max_counts[scenario]
                if max_counts[scenario]
                else np.zeros_like(counts, dtype=float)
            )
            bin_colors = half_cmap(color_norm(relative_counts))

            ax.bar(
                sample_times,
                upper - lower,
                bottom=lower,
                width=TIME_BIN_WIDTH,
                align="center",
                color=bin_colors,
                edgecolor="none",
                label=f"{scenario.capitalize()} 90% envelope",
                zorder=2 if scenario == "worst" else 1,
            )
            ax.step(
                sample_times,
                median,
                where="mid",
                color=color,
                lw=2,
                label=f"{scenario.capitalize()} median",
                zorder=4 if scenario == "worst" else 3,
            )

        ax.grid(alpha=0.6)
        ax.legend(fontsize=10)
        ax.set_title(title)
        ax.set_xlabel("Time since merger [day]")

    axes[0].set_ylabel(f"{color_label} [mag]")
    fig.suptitle(
        f"{color_label} — {comparison.replace('_', ' ')} — {suffix}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.23, 1, 0.93])
    colorbar_width = 0.36 if len(colorbars) == 1 else 0.4
    colorbar_gap = 0.04
    total_colorbar_width = (
        len(colorbars) * colorbar_width + (len(colorbars) - 1) * colorbar_gap
    )
    colorbar_start = (1 - total_colorbar_width) / 2
    for index, (scenario, mappable) in enumerate(colorbars.items()):
        colorbar_ax = fig.add_axes(
            [
                colorbar_start + index * (colorbar_width + colorbar_gap),
                0.11,
                colorbar_width,
                0.025,
            ]
        )
        colorbar = fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label(f"Relative number of contributing curves ({scenario})")

    output_file = f"{PLOT_DIR}/{filename_prefix}_{comparison}_magcut_{suffix}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.close(fig)


def plot_color(color_data, color_label, filename_prefix):
    """Save best, worst, and combined-cut plots for each distance and all distances."""
    for comparison in COMPARISONS:
        for dl in DLs:
            plot_comparison(
                color_data,
                color_label,
                filename_prefix,
                comparison,
                [dl],
                f"dl_{dl:g}Mpc",
            )

        plot_comparison(
            color_data,
            color_label,
            filename_prefix,
            comparison,
            DLs,
            "all_dls",
        )


os.makedirs(PLOT_DIR, exist_ok=True)

plot_color(GR, "g − r", "gr_col")
plot_color(RI, "r − i", "ri_col")
plot_color(GI, "g − i", "gi_col")