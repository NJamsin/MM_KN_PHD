from nmma.em.model import FiestaKilonovaModel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

OUTDIR = "/home/liteul/phd/NSBH/lightBH"
PLOT_DIR = f"{OUTDIR}/plot"

MAG_LIMITS = {
    "best": {"lsstg": 24.8, "lsstr": 24.4, "lssti": 23.9},
    "worst": {"lsstg": 24.2, "lsstr": 23.7, "lssti": 23.4},
}
ENVELOPE_PERCENTILES = [5, 50, 95]

fiesta_kn_name = "Bu2019nsbh_MLP"

filters = ["lsstg", "lsstr", "lssti"]

fiesta_kn_model = FiestaKilonovaModel(model=fiesta_kn_name, filters=filters)

sample_times = np.linspace(0.1,19,190)

seed_S_file = f"{OUTDIR}/SLy/seed.csv"
seed_H_file = f"{OUTDIR}/H4/seed.csv"

seed_S = pd.read_csv(seed_S_file, sep=',')
seed_H = pd.read_csv(seed_H_file, sep=',')

params_name = [
    "log10_mej_dyn",
    "log10_mej_wind", 
]

DLs = np.linspace(40, 400, 3)

GR = {}
RI = {}

seed = {"H4": seed_H, "SLy": seed_S}

for eos in ["H4", "SLy"]:
    print(f"Computing color for EOS {eos}")
    GR[eos] = {}
    RI[eos] = {}
    for incl, theta in zip(["on", "off"], [0,90]):
        print(f"Incl {incl}")
        GR[eos][incl] = {scenario: [] for scenario in MAG_LIMITS}
        RI[eos][incl] = {scenario: [] for scenario in MAG_LIMITS}
        for dl in DLs:
            for i in tqdm(range(len(seed[eos]['redshift']))):
                parameters = {
                    name: seed[eos][name][i] for name in params_name
                }
                parameters['KNtheta'] = theta
                parameters['luminosity_distance'] = dl

                mag = fiesta_kn_model.generate_lightcurve(sample_times, parameters)
                for scenario, limits in MAG_LIMITS.items():
                    detected_gr = (
                        (mag["lsstg"] <= limits["lsstg"])
                        & (mag["lsstr"] <= limits["lsstr"])
                    )
                    detected_ri = (
                        (mag["lsstr"] <= limits["lsstr"])
                        & (mag["lssti"] <= limits["lssti"])
                    )
                    GR[eos][incl][scenario].append(
                        np.where(detected_gr, mag["lsstg"] - mag["lsstr"], np.nan)
                    )
                    RI[eos][incl][scenario].append(
                        np.where(detected_ri, mag["lsstr"] - mag["lssti"], np.nan)
                    )

def summarize_curves(curves):
    """Compute the requested percentiles while allowing undetected epochs."""
    values = np.stack(curves)
    summary = np.full((len(ENVELOPE_PERCENTILES), values.shape[1]), np.nan)
    for time_index in range(values.shape[1]):
        detected_values = values[:, time_index]
        detected_values = detected_values[np.isfinite(detected_values)]
        if detected_values.size:
            summary[:, time_index] = np.percentile(
                detected_values, ENVELOPE_PERCENTILES
            )
    return summary


def plot_color(color_data, color_label, filename_prefix):
    """Save separate best/worst plots and a plot comparing both magnitude cuts."""
    for comparison in ["best", "worst", "best_worst"]:
        fig, axes = plt.subplots(
            figsize=(18, 14), ncols=2, nrows=2, sharey=True, sharex=True
        )

        for i, eos in enumerate(["H4", "SLy"]):
            for j, (incl, title) in enumerate(
                zip(["on", "off"], ["Face-on", "Edge-on"])
            ):
                ax = axes[i, j]
                scenarios = (
                    ["best", "worst"] if comparison == "best_worst" else [comparison]
                )
                for scenario in scenarios:
                    if scenario == "worst":
                        kwargs = {"zorder": 10}
                    else:
                        kwargs = {}
                    lower, median, upper = summarize_curves(
                        color_data[eos][incl][scenario]
                    )
                    line, = ax.plot(
                        sample_times,
                        median,
                        lw=2,
                        label=f"{scenario.capitalize()} median",
                        **kwargs
                    )
                    ax.fill_between(
                        sample_times,
                        lower,
                        upper,
                        color=line.get_color(),
                        alpha=0.2,
                        label=f"{scenario.capitalize()} envelope (90% CI)",
                        **kwargs
                    )

                ax.grid(alpha=0.6)
                ax.legend(fontsize=11)
                ax.set_title(f"{eos} - {title}")
                if j == 0:
                    ax.set_ylabel(f"{color_label} [mag]")
                if i == 1:
                    ax.set_xlabel("Time since merger [day]")

        fig.tight_layout()
        output_file = f"{PLOT_DIR}/{filename_prefix}_{comparison}_magcut.png"
        fig.savefig(output_file, dpi=300)
        print(f"Saved {output_file}")
        plt.close(fig)


os.makedirs(PLOT_DIR, exist_ok=True)

plot_color(GR, "g − r", "gr_col")
plot_color(RI, "r − i", "ri_col")