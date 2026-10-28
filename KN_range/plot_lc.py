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

fiesta_kn_name = "Bu2019nsbh_MLP"

filters = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]

limit_best = {"lsstu": 23.7,
              "lsstg": 24.8,
              "lsstr": 24.4, 
              "lssti": 23.9, 
              "lsstz": 23.3, 
              "lssty": 22.3}
limit_worst = {"lsstu": 23.1,
               "lsstg": 24.2, 
               "lsstr": 23.7, 
               "lssti": 23.4, 
               "lsstz": 22.8, 
               "lssty": 21.8}

fiesta_kn_model = FiestaKilonovaModel(model=fiesta_kn_name, filters=filters)

sample_times = np.logspace(np.log10(0.5), np.log10(19), 35)

seed_H_file = f"{OUTDIR}/seed.csv"

seed = pd.read_csv(seed_H_file, sep=',')

params_name = [
    "log10_mej_dyn",
    "log10_mej_wind", 
]

DLs = [40,150,300,400, "seed"]

for incl, theta in zip(["on", "off"], [0,90]):
    print(f"Incl {incl}")
    for dl in DLs:
        if dl == "seed" and incl == "off":
            continue
        print(f"Dist {dl}")
        fig, ax = plt.subplots(len(filters), 1, figsize=(10,3.25*len(filters)), sharex=True)

        for i in tqdm(range(len(seed['redshift']))):
            parameters = {
                name: seed[name][i] for name in params_name
            }
            if dl == "seed":
                parameters["KNtheta"] = seed["KNtheta"][i]
                parameters["luminosity_distance"] = seed["luminosity_distance"][i]
            else:
                parameters['KNtheta'] = theta
                parameters['luminosity_distance'] = dl

            mag = fiesta_kn_model.generate_lightcurve(sample_times, parameters)
            mag = abs_to_app_mag(mag, parameters['luminosity_distance'])

            for filter_idx, filter_name in enumerate(filters):
                magnitude = mag[filter_name]
                ax[filter_idx].plot(sample_times, magnitude, alpha=0.7)
        for filter_idx, filter_name in enumerate(filters):
            ax[filter_idx].axhline(limit_best[filter_name], color="green", label=f"{filter_name} best cut")
            ax[filter_idx].axhline(limit_worst[filter_name], color="red", label=f"{filter_name} worst cut")
            ax[filter_idx].set_ylabel(f'{filter_name} (AB mag)', fontsize=12)
            ax[filter_idx].set_ylim(None, limit_best[filter_name]+2)
            ax[filter_idx].set_xscale('log')
            ax[filter_idx].invert_yaxis()
            ax[filter_idx].legend(loc="lower left")
        ax[-1].set_xlabel('Time since merger (days)', fontsize=12)
        if dl=="seed":
            ax[0].set_title(rf'NSBH light curves from binary system ({len(seed['redshift'])} samples) - $\theta$ & $D_L$ from seed file',
                                        fontsize=14, fontweight='bold')
        else:
            ax[0].set_title(rf'NSBH light curves from binary system ({len(seed['redshift'])} samples) - $\theta$={theta} $D_L$={dl}',
                            fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/lc_{theta}_{dl}.png", dpi=300)
        plt.close()