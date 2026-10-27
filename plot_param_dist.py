import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


OUTDIR = "/home/liteul/phd/NSBH/lightBH"

seed_file_H = f"{OUTDIR}/H4/seed.csv"
seed_H = pd.read_csv(seed_file_H, delimiter=',')

seed_file_S = f"{OUTDIR}/SLy/seed.csv"
seed_S = pd.read_csv(seed_file_H, delimiter=',')

# Liste des paramètres d'intérêt
param_names = [
    "mass_1", 
    "mass_2", 
    "mass_ratio",
    "chi_bh", 
    "zeta", 
    "log10_mej_dyn",
    "log10_mej_wind", 
]
params_latex = [
    r"$m_1$",
    r"$m_2$",
    r"$q$",
    r"$\chi_{\rm{BH}}$",
    r"$\zeta$",
    r"$\log_{10} M_{ej}^{dyn}$", 
    r"$\log_{10} M_{ej}^{wind}$",
]

# Filtrer sur les colonnes réellement présentes dans le fichier
cols_to_plot = [col for col in param_names if col in seed_H.columns]

# ----------------------------------------------------
# 2. Configuration de la grille de sous-graphiques
# ----------------------------------------------------
n_params = len(cols_to_plot)
n_cols = 3
n_rows = math.ceil(n_params / n_cols)

for seed, eos in zip([seed_H, seed_S], ["H4", "SLy"]):

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).flatten()

    # ----------------------------------------------------
    # 3. Tracé des histogrammes
    # ----------------------------------------------------
    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        if col == "mass_ratio":
            val1 = seed["mass_1"].dropna()
            val2 = seed["mass_2"].dropna()
            values = val1/val2
        else:
            values = seed[col].dropna()

        # Histogramme normalisé (densité)
        ax.hist(values, bins=35, density=True, alpha=0.65, color="#1f77b4", edgecolor="black", linewidth=0.6)

        # Médiane et quantiles (16% et 84% pour l'intervalle 1-sigma)
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        ax.axvline(q50, color="crimson", linestyle="-", linewidth=1.5, label=f"Median: {q50:.2g}")
        ax.axvline(q16, color="crimson", linestyle="--", linewidth=1.0)
        ax.axvline(q84, color="crimson", linestyle="--", linewidth=1.0)

        ax.set_title(params_latex[idx], fontsize=12, fontweight="bold")
        ax.set_ylabel("Density (AU)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    # Masquer les sous-graphiques vides résiduels
    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Parameters distributions - {eos} KN", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    # Sauvegarde
    output_png = f"{OUTDIR}/plot/seed_distributions_{eos}.png"
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Distributions 1D sauvegardées dans '{output_png}'.")