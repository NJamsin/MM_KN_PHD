import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner


OUTDIR = "/home/liteul/phd/NSBH/KN_range/seed_all"

seed_files = {
    "H4": f"{OUTDIR}/seed.csv",
    #"SLy": f"{OUTDIR}/SLy/seed.csv",
}

param_names = [
    "mass_1",
    "mass_2",
    "mass_ratio",
    "chirp_mass",
    "chi_bh",
    "zeta",
    "log10_mej_dyn",
    "log10_mej_wind",
]
params_latex = {
    "mass_1": r"$m_1$",
    "mass_2": r"$m_2$",
    "mass_ratio": r"$q$",
    "chirp_mass": r"$\mathcal{M}$",
    "chi_bh": r"$\chi_{\rm{BH}}$",
    "zeta": r"$\zeta$",
    "log10_mej_dyn": r"$\log_{10} M_{ej}^{dyn}$",
    "log10_mej_wind": r"$\log_{10} M_{ej}^{wind}$",
}

for eos, seed_file in seed_files.items():
    seed_raw = pd.read_csv(seed_file)

    mask = seed_raw["KN"].astype(bool)
    seed_raw["mass_ratio"] = seed_raw["mass_2"] / seed_raw["mass_1"]
    seed_raw["chirp_mass"] = (
        seed_raw["mass_1"] * seed_raw["mass_2"]
    ) ** (3 / 5) / (seed_raw["mass_1"] + seed_raw["mass_2"]) ** (1 / 5)

    n_params = len(param_names)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.asarray(axes).flatten()
    groups = [
        (seed_raw.loc[mask], "#1f77b4", "KN generated"),
        (seed_raw.loc[~mask], "#d43200", "No KN"),
    ]
    cols_to_plot = [name for name in param_names if name in seed_raw.columns]

    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        finite_groups = []
        for group_seed, color, label in reversed(groups):
            values = pd.to_numeric(group_seed[col], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            finite_groups.append((values, color, label))

        finite_values = [
            values for values, _, _ in finite_groups if values.size
        ]
        if finite_values:
            # Use shared bin edges so the two normalized distributions are comparable.
            bin_edges = np.histogram_bin_edges(
                np.concatenate(finite_values), bins=35
            )
            for values, color, label in finite_groups:
                if not values.size:
                    continue
                ax.hist(
                    values,
                    bins=bin_edges,
                    density=True,
                    alpha=0.45,
                    color=color,
                    edgecolor=color,
                    linewidth=0.8,
                    label=f"{label} (n={values.size:,})",
                )

                q16, q50, q84 = np.percentile(values, [16, 50, 84])
                ax.axvline(
                    q50,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    label=f"{label} median: {q50:.2g}",
                )
                ax.axvline(q16, color=color, linestyle="--", linewidth=1.0)
                ax.axvline(q84, color=color, linestyle="--", linewidth=1.0)
        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(params_latex[col], fontsize=12, fontweight="bold")
        ax.set_ylabel("Density (AU)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper right")

    for idx in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(
        f"Parameter distributions — {eos}: KN generated vs no KN",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    output_png = f"{OUTDIR}/plot/seed_distributions_{eos}_chirp_mass.png"
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Distributions 1D sauvegardées dans '{output_png}'.")

    corner_groups = []
    for group_seed, color, label in groups:
        samples = (
            group_seed[cols_to_plot]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        samples = samples[np.isfinite(samples).all(axis=1)]
        if not samples.size:
            raise ValueError(
                f"No complete finite samples available for {label} ({eos}) corner plot."
            )
        corner_groups.append((samples, color, label))

    combined_samples = np.concatenate(
        [samples for samples, _, _ in corner_groups], axis=0
    )
    corner_ranges = [
        (float(np.min(combined_samples[:, idx])), float(np.max(combined_samples[:, idx])))
        for idx in range(len(cols_to_plot))
    ]
    corner_plot_groups = [corner_groups[1], corner_groups[0]]
    corner_fig = corner.corner(
        corner_plot_groups[0][0],
        labels=[params_latex[col] for col in cols_to_plot],
        range=corner_ranges,
        bins=35,
        color=corner_plot_groups[0][1],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        plot_datapoints=False,
        fill_contours=True,
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
    )
    corner.corner(
        corner_plot_groups[1][0],
        labels=[params_latex[col] for col in cols_to_plot],
        range=corner_ranges,
        bins=35,
        color=corner_plot_groups[1][1],
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        fill_contours=True,
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
        fig=corner_fig,
    )
    n_corner_params = len(cols_to_plot)
    for parameter_idx in range(n_corner_params):
        diagonal_ax = corner_fig.axes[
            parameter_idx * n_corner_params + parameter_idx
        ]
        for group_idx, (samples, color, label) in enumerate(corner_plot_groups):
            q16, median, q84 = np.percentile(
                samples[:, parameter_idx], [16, 50, 84]
            )
            title = (
                f"{label}: ${median:.2g}_{{-{median - q16:.2g}}}"
                f"^{{+{q84 - median:.2g}}}$"
            )
            diagonal_ax.text(
                0.5,
                1.08 + group_idx * 0.14,
                title,
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                transform=diagonal_ax.transAxes,
                clip_on=False,
            )

    corner_fig.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=2, label=f"{label} (n={len(samples):,})")
            for samples, color, label in corner_groups
        ]
        + [
            plt.Line2D(
                [0], [0], color="black", lw=1.5, label="Median (solid)"
            ),
            plt.Line2D(
                [0], [0], color="black", lw=1.0, linestyle="--",
                label="16th–84th percentiles (dashed)",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )
    corner_fig.suptitle(
        f"Corner distributions — {eos}: KN generated vs no KN",
        fontsize=14,
        fontweight="bold",
    )
    corner_fig.subplots_adjust(top=0.88)
    corner_png = f"{OUTDIR}/plot/seed_corner_{eos}_chirp_mass.png"
    corner_fig.savefig(corner_png, dpi=300, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Corner plot sauvegardé dans '{corner_png}'.")