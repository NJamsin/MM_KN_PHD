from nmma.em.model import FiestaKilonovaModel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from peket.kn_side.utils import abs_to_app_mag, chirp_mass

OUTDIR = "/home/liteul/phd/NSBH/KN_range/H4_chi07"
PLOT_DIR = f"{OUTDIR}/plot"
DISTANCE_MPC = 350
CHIRP_MASS_BINS = 10

MAG_LIMITS = {
    "best": {
        "lsstu": 23.7,
        "lsstg": 24.8,
        "lsstr": 24.4,
        "lssti": 23.9,
        "lsstz": 23.3,
        "lssty": 22.3,
    },
    "worst": {
        "lsstu": 23.1,
        "lsstg": 24.2,
        "lsstr": 23.7,
        "lssti": 23.4,
        "lsstz": 22.8,
        "lssty": 21.8,
    },
}

fiesta_kn_name = "Bu2019nsbh_MLP"

filters = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]

fiesta_kn_model = FiestaKilonovaModel(model=fiesta_kn_name, filters=filters)

sample_times = np.arange(0.5, 5.5, 2)

seed_H_file = f"{OUTDIR}/seed.csv"

seed = pd.read_csv(seed_H_file, sep=",")

params_name = [
    "log10_mej_dyn",
    "log10_mej_wind",
]

chirp_masses = np.asarray(
    [
        chirp_mass(mass_1, mass_2)
        for mass_1, mass_2 in zip(seed["mass_1"], seed["mass_2"])
    ]
)
detections = {}

for incl, theta in zip(["on", "off"], [0, 90]):
    print(f"Incl {incl}")
    detections[incl] = {scenario: [] for scenario in MAG_LIMITS}
    for i in tqdm(range(len(seed["redshift"]))):
        parameters = {name: seed[name][i] for name in params_name}
        parameters["KNtheta"] = theta
        parameters["luminosity_distance"] = DISTANCE_MPC
        mag = fiesta_kn_model.generate_lightcurve(sample_times, parameters)
        mag = abs_to_app_mag(mag, DISTANCE_MPC)

        for scenario, limits in MAG_LIMITS.items():
            detected_filters = np.stack(
                [
                    mag[filter_name] <= limit
                    for filter_name, limit in limits.items()
                ]
            )
            epochs_with_multiple_detections = (
                detected_filters.sum(axis=0) >= 2
            )
            detections[incl][scenario].append(
                np.count_nonzero(epochs_with_multiple_detections) >= 3
            )


def plot_detections_vs_chirp_mass():
    """Plot per-bin detection efficiency at the fixed distance."""
    bin_edges = np.histogram_bin_edges(chirp_masses, bins=CHIRP_MASS_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    number_of_series = len(detections) * len(MAG_LIMITS)
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.8 / number_of_series
    samples_per_bin, _ = np.histogram(chirp_masses, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(12, 6))
    series_index = 0
    colors = plt.colormaps["tab10"].colors
    for incl, incl_label in [("on", "Face-on"), ("off", "Edge-on")]:
        for scenario in MAG_LIMITS:
            detected = np.asarray(detections[incl][scenario], dtype=bool)
            detected_per_bin, _ = np.histogram(
                chirp_masses[detected], bins=bin_edges
            )
            efficiency = np.divide(
                detected_per_bin,
                samples_per_bin,
                out=np.full(detected_per_bin.shape, np.nan, dtype=float),
                where=samples_per_bin > 0,
            )
            offset = (series_index - (number_of_series - 1) / 2) * bar_width
            ax.bar(
                bin_centers + offset,
                efficiency,
                width=bar_width,
                label=f"{incl_label} — {scenario.capitalize()}",
                color=colors[series_index],
            )
            series_index += 1

    ax.set_xlabel(r"Chirp mass [$M_\odot$]")
    ax.set_ylabel("Detected events / events in bin")
    ax.set_title(f"Detection efficiency vs chirp mass at {DISTANCE_MPC} Mpc")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(bin_centers, [f"{center:.2f}" for center in bin_centers])
    ax.grid(axis="y", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_file = f"{PLOT_DIR}/detections_vs_mchirp_{DISTANCE_MPC}Mpc.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.close(fig)


os.makedirs(PLOT_DIR, exist_ok=True)
plot_detections_vs_chirp_mass()
