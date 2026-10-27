from nmma.em.model import FiestaKilonovaModel
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

OUTDIR = "/home/liteul/phd/NSBH/color_m_cut/all_chi"

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

DLs = np.arange(40, 440, 50)

GR = {}
RI = {}

seed = {"H4": seed_H, "SLy": seed_S}

for eos in ["H4", "SLy"]:
    print(f"Computing color for EOS {eos}")
    GR[eos] = {}
    RI[eos] = {}
    for incl, theta in zip(["on", "off"], [0,90]):
        print(f"Incl {incl}")
        GR[eos][incl] = []
        RI[eos][incl] = []
        for dl in DLs:
            for i in tqdm(range(len(seed[eos]['redshift']))):
                parameters = {
                    name: seed[eos][name][i] for name in params_name
                }
                parameters['KNtheta'] = theta
                parameters['luminosity_distance'] = dl

                mag = fiesta_kn_model.generate_lightcurve(sample_times, parameters)

                GR[eos][incl].append(mag["lsstg"] - mag["lsstr"])
                RI[eos][incl].append(mag["lsstr"] - mag["lssti"])

# PLOT 

fig, axes = plt.subplots(figsize=(18, 14), ncols=2, nrows=2, sharey=True, sharex=True)
fig2, axes2 = plt.subplots(figsize=(18, 14), ncols=2, nrows=2, sharey=True, sharex=True)

for i, eos in enumerate(["H4", "SLy"]):
    print(f"plot for EOS {eos}")
    for j, (incl, tit) in enumerate(zip(["on", "off"], ["Face-on", "Edge-on"])):
        print(i, j)
        ax = axes[i,j]
        ax2 = axes2[i, j]

        gr_curves = np.stack(GR[eos][incl])  
        ri_curves = np.stack(RI[eos][incl])

        p05, median, p95 = np.nanpercentile(gr_curves, [5, 50, 95], axis=0)

        p0, median2, p9 = np.nanpercentile(ri_curves, [5, 50, 95], axis=0)

        ax.fill_between(sample_times, p05, p95, alpha=0.25, label=r"Envelope (90% CI)")
        ax.plot(sample_times, median, color="C0", lw=2, label="Median")
        ax2.fill_between(sample_times, p0, p9, alpha=0.25, label=r"Envelope (90% CI)")
        ax2.plot(sample_times, median2, color="C0", lw=2, label="Median")
        ax.grid(alpha=0.6)
        ax.legend(fontsize=13)
        ax.set_title(f"{eos} - {tit}")
        ax2.grid(alpha=0.6)
        ax2.legend(fontsize=13)
        ax2.set_title(f"{eos} - {tit}")

for i in [0,1]:
    axes[i,0].set_ylabel("g − r [mag]")
    axes2[i,0].set_ylabel("r − i [mag]")

    axes[1,i].set_xlabel("Time since merger [day]")
    axes2[1,i].set_xlabel("Time since merger [day]")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot/gr_col.png", dpi=300)
fig2.tight_layout()
fig2.savefig(f"{OUTDIR}/plot/ri_col.png", dpi=300)
plt.show()