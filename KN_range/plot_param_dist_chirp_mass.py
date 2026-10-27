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
    "log10_mej"
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
    "log10_mej": r"$\log_{10} M_{ej}$",
}

for eos, seed_file in seed_files.items():
    seed = pd.read_csv(seed_file)

    mass_samples = seed[["mass_1", "mass_2"]].dropna()
    mass_1 = mass_samples["mass_1"]
    mass_2 = mass_samples["mass_2"]
    seed["mass_ratio"] = mass_2 / mass_1
    seed["chirp_mass"] = (mass_1 * mass_2) ** (3 / 5) / (mass_1 + mass_2) ** (1 / 5)
    seed["log10_mej"] = np.log10(10**seed["log10_mej_dyn"] + 10**seed["log10_mej_wind"])

    #fraction of lc with mej > 0.3 msun
    prop_mask = seed["log10_mej"] >= np.log10(0.3)
    frac = prop_mask.sum()
    print(f"The fraction of lc having an ejecta mass greater or equal to 0.3 Msun is {frac/len(prop_mask)}")

    cols_to_plot = [name for name in param_names if name in seed.columns]
    n_params = len(cols_to_plot)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.asarray(axes).flatten()

    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        values = seed[col].apply(pd.to_numeric, errors="coerce").dropna()
        values = values[np.isfinite(values)]
        ax.hist(
            values,
            bins=35,
            density=True,
            alpha=0.65,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.6,
        )

        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        ax.axvline(q50, color="crimson", linestyle="-", linewidth=1.5, label=f"Median: {q50:.2g}")
        ax.axvline(q16, color="crimson", linestyle="--", linewidth=1.0)
        ax.axvline(q84, color="crimson", linestyle="--", linewidth=1.0)

        ax.set_title(params_latex[col], fontsize=12, fontweight="bold")
        ax.set_ylabel("Density (AU)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Parameters distributions - {eos} KN", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()

    output_png = f"{OUTDIR}/plot/seed_distributions_{eos}_chirp_mass.png"
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Distributions 1D sauvegardées dans '{output_png}'.")

    corner_data = seed[cols_to_plot].apply(pd.to_numeric, errors="coerce").dropna()
    corner_fig = corner.corner(
        corner_data.to_numpy(),
        labels=[params_latex[col] for col in cols_to_plot],
        bins=35,
        color="#1f77b4",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
    )
    corner_fig.suptitle(f"Bivariate parameter distributions - {eos} KN", fontsize=14, fontweight="bold")
    #corner_fig.tight_layout()

    corner_png = f"{OUTDIR}/plot/seed_corner_{eos}_chirp_mass.png"
    corner_fig.savefig(corner_png, dpi=300, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Corner plot sauvegardé dans '{corner_png}'.")
