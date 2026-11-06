# prior and stuff (only on distance, redshift in converted internally by nmma)
import bilby.core.prior as prior
from bilby.gw.prior import UniformComovingVolume
from astropy.cosmology import Planck18, z_at_value
from bilby.core.utils.random import seed
from nmma.em.model import FiestaKilonovaModel

fiesta_kn_name = "Bu2019nsbh_MLP"
filters = ["lsstu", "lsstg", "lsstr", "lssti", "lssty", "lsstz"]
fiesta_kn_model = FiestaKilonovaModel(model=fiesta_kn_name, filters=filters)

seed(170817)

OUT_DIR = '/home/liteul/phd/NSBH/KN_range/test_19_e'
n_samples = 1000000

cosmo = Planck18
priors = prior.PriorDict()

priors["mass_1"] = prior.Uniform(
    3, 10, "mass_1", latex_label=r"$m_1$"
)
priors["mass_2"] = prior.Uniform(
    1, 2.33, "mass_2", latex_label=r"$m_2$"
)
priors["KNtheta"] = prior.Uniform(
    0.0, 90.0, "KNtheta", latex_label=r"$\theta_{\rm{KN}}$"
)
priors["luminosity_distance"] = UniformComovingVolume(
    40, 500, cosmo, "luminosity_distance", latex_label=r"$D_L$", unit="Mpc"
)
priors["chi_bh"] = prior.Uniform(
    0., 0.7, "chi_bh", latex_label=r"$\chi_{\rm{bh}}$"
)

samples = priors.sample(size=n_samples)

import pandas as pd

H4 = pd.read_csv('/home/liteul/phd/NSBH/EOS/H4.mr', header=0, sep=r'\s+')

log10_mej_dyn_b = (-2.0, -1.0457574905606752)
log10_mej_wind_b = (-2.0, -1.0457574905606752)

import os, csv

columns = [
    "mass_1", "mass_2", "chi_bh", "zeta", "KNtheta", "log10_mej_dyn",
    "log10_mej_wind", "luminosity_distance", 
    "redshift", "object_id",
    "type", "version", "RA", "DEC", "KN",
    "t_det_start", "t_det_end",  # Nouvelles colonnes pour la période
]

def append_rows(seed_file, rows):
    write_header = not os.path.exists(seed_file) or os.path.getsize(seed_file) == 0
    with open(seed_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

import importlib
import warnings
from pathlib import Path
import numpy as np
import peket.kn_side.utils as utils
import astropy.units as u
from tqdm import tqdm

# On étend sample_times jusqu'à 2 jours (par ex. 20 points entre 0 et 2.0 jours)
sample_times = np.linspace(0.0, 0.1, 5)

m_ej_dyns = {}
m_disks = {}

for eos, name in zip([H4], ["H4"]):
    print(f"Generating {len(samples['mass_1'])} lc for eos {name}")
    seed_file = f"{OUT_DIR}/seed.csv"
    os.makedirs(os.path.dirname(seed_file), exist_ok=True)
    Path(seed_file).unlink(missing_ok=True)
    
    M_TOV_err = 0
    M_dyn_0_err = 0
    M_dyn_out_err = 0
    M_disk_o_err = 0
    M_wind_out_err = 0

    m_ej_dyns[name] = []
    m_disks[name] = []

    for i in tqdm(range(len(samples['mass_1']))):
        KN_fiesta = True
        params = {
            "KNtheta": samples["KNtheta"][i],
            "luminosity_distance": samples["luminosity_distance"][i],
        }
        chi_bh = samples["chi_bh"][i]
        mass_1 = samples["mass_1"][i]
        mass_2 = samples["mass_2"][i]
        zeta = utils.zeta_nsbh(mass_1/mass_2)
        mass_max = eos["Mass"].max()
        if mass_2 > mass_max:
            M_TOV_err += 1
            warnings.warn(f"Neutron Star mass larger than M_TOV for eos {name}")
            continue
        C2 = utils.NS_compactness(mass_2, np.interp(mass_2, eos['Mass'], eos['Radius']))
        mej_dyn = utils.dynamic_mass_fitting(
                                    mass_1_source=mass_1,
                                    mass_2_source=mass_2,
                                    compactness_2=C2,
                                    chi_bh=chi_bh)
        m_disk = utils.remnant_disk_mass_fitting(
                                    mass_1_source=mass_1,
                                    mass_2_source=mass_2,
                                    compactness_2=C2,
                                    chi_bh=chi_bh)
        m_disks[name].append(m_disk)
        m_ej_dyns[name].append(mej_dyn)
        if mej_dyn == 0.0:
            M_dyn_0_err += 1
            KN_fiesta = False
            continue
        if not log10_mej_dyn_b[0] <= np.log10(mej_dyn) <= log10_mej_dyn_b[1]:
            M_dyn_out_err += 1
            KN_fiesta = False
            continue
        params["log10_mej_dyn"] = np.log10(mej_dyn)

        if m_disk <= 0.0:
            M_disk_o_err += 1
            KN_fiesta = False 
            continue
        mej_wind = m_disk * zeta
        if not log10_mej_wind_b[0] <= np.log10(mej_wind) <= log10_mej_wind_b[1]:
            M_wind_out_err += 1
            KN_fiesta = False
            continue
        params["log10_mej_wind"] = np.log10(mej_wind)

        # Génération de la courbe de lumière 
        mag_m = fiesta_kn_model.generate_lightcurve(sample_times, params)
        mag_m = utils.abs_to_app_mag(mag_m, samples["luminosity_distance"][i])

        # Vérification de la détection (plus lumineux que 19 mag -> magnitude < 19)
        detected_times = []
        for idx, t in enumerate(sample_times):
            is_detected_at_t = False
            for filt in filters:
                if mag_m[filt][idx] < 19:  # < 19 = plus lumineux que 19 mag
                    is_detected_at_t = True
                    break
            if is_detected_at_t:
                detected_times.append(t)
        
        # Si aucune détection sur toute la période [0, 2] jours, on ignore
        if not detected_times:
            continue
            
        t_det_start = detected_times[0]
        t_det_end = detected_times[-1]

        # Enregistrement des paramètres et de la période de détection dans seed.csv
        row = {
                "mass_1": mass_1, 
                "mass_2": mass_2, 
                "chi_bh": chi_bh, 
                "zeta": zeta, 
                "KNtheta": params["KNtheta"], 
                "log10_mej_dyn": params["log10_mej_dyn"],
                "log10_mej_wind": params["log10_mej_wind"], 
                "luminosity_distance": params["luminosity_distance"], 
                "redshift" : float(z_at_value(cosmo.luminosity_distance, params['luminosity_distance'] * u.Mpc)),
                "object_id": i,
                "type": "NSBH-KN",
                "version": "Bu2019nsbh_MLP",
                "RA": " ",
                "DEC": " ",
                "KN": f"{KN_fiesta}",
                "t_det_start": t_det_start,
                "t_det_end": t_det_end,
            }
        append_rows(seed_file, [row])       

    print(f"Lc generation complete for eos {name}\n")
    print(f"{M_TOV_err} lc not generated due to M2 being above M_TOV for eos {name} ({M_TOV_err/len(samples['mass_1']) * 100}%)")
    print(f"{M_dyn_0_err} lc not generated due to dynamical ejecta mass being 0 ({M_dyn_0_err/len(samples['mass_1']) * 100}%)")
    print(f"{M_dyn_out_err} lc not generated due to dynamical ejecta being outside parameter space ({M_dyn_out_err/len(samples['mass_1']) * 100}%)")
    print(f"{M_disk_o_err} lc not generated due to disk mass being 0 ({M_disk_o_err/len(samples['mass_1']) * 100}%)")
    print(f"{M_wind_out_err} lc not generated due to wind ejecta being outside parameter space ({M_wind_out_err/len(samples['mass_1']) * 100}%)")
    print(f"Generated {len(samples['mass_1'])-M_TOV_err-M_dyn_0_err-M_dyn_out_err-M_disk_o_err-M_wind_out_err} light curves out of {len(samples['mass_1'])} ({(len(samples['mass_1'])-M_TOV_err-M_dyn_0_err-M_dyn_out_err-M_disk_o_err-M_wind_out_err)/len(samples['mass_1'])}%)")