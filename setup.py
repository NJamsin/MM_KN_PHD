import numpy as np
import subprocess
import shlex
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
from ligo.skymap.io import fits
import json


PALOMAR = EarthLocation.of_site("Palomar")

def generate_and_plot_fake_skymap(center_coord, sigma_deg=55.0, out_file='fake_lvk_alert.multiorder.fits'):
    """
    Génère une fausse carte de localisation GW au format MOC FITS et l'affiche.
    
    Paramètres:
    - center_coord : Objet astropy.coordinates.SkyCoord définissant le centre.
    - sigma_deg : Incertitude spatiale (1-sigma) en degrés.
    - out_file : Nom du fichier de sortie.
    """
    
    # 1. Paramètres de la grille healpy
    nside = 512
    npix = hp.nside2npix(nside)
    
    # 2. Conversion des coordonnées pour healpy (colatitude theta, longitude phi en radians)
    theta_center = 0.5 * np.pi - center_coord.dec.rad
    phi_center = center_coord.ra.rad
    vec_center = hp.ang2vec(theta_center, phi_center)
    
    # 3. Calcul des distances angulaires sur la sphère
    vecs = hp.pix2vec(nside, np.arange(npix))
    cos_angles = np.clip(np.dot(vec_center, vecs), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    # 4. Probabilité (Gaussienne)
    sigma_rad = np.radians(sigma_deg)
    prob = np.exp(-0.5 * (angles / sigma_rad)**2)
    prob /= prob.sum()
    
    # 5. Ajout des paramètres de distance 3D factices (en Mpc)
    distmean = 150.0 
    diststd = 40.0
    distmu = np.full(npix, distmean)
    distsigma = np.full(npix, diststd)
    distnorm = np.full(npix, 1.0)
    
    # 6. Sauvegarde au format MOC .fits
    fits.write_sky_map(
        out_file, 
        (prob, distmu, distsigma, distnorm), 
        nest=False, 
        moc=True,
        gps_time=1380000000.0, # Temps GPS arbitraire
        creator='Custom Python Script'
    )
    print(f"Skymap MOC générée avec succès : {out_file}")

def lst_deg(t):
    return t.sidereal_time("apparent", longitude=PALOMAR.lon).deg

def sky_direction(body):
    """RA/Dec vus de la Terre, sans distance.

    get_sun / get_body renvoient du GCRS avec distance : un .icrs donnerait une
    direction barycentrique (fausse). On ne garde donc que (ra, dec).
    """
    return SkyCoord(ra=body.ra, dec=body.dec, frame="icrs")

def _run(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def night_bounds(date_utc):
    """Début/fin de nuit astronomique (soleil < -18°) entre 00h et 16h UTC de date_utc."""
    t = Time(date_utc + "T00:00:00") + np.arange(0, 16 * 60) * u.min
    alt = get_sun(t).transform_to(AltAz(obstime=t, location=PALOMAR)).alt.deg
    dark = np.where(alt < -18)[0]
    return t[dark[0]], t[dark[-1]]


def build_tests():
    tests = []

    # Nuit de référence : 7 nov 2026 (heure locale du 6 au 7)
    start, end = night_bounds("2026-11-07")
    mid = start + (end - start) / 2
    ra_mid = lst_deg(mid)

    #tests.append(dict(name="T0_control", ra=ra_mid, dec=+20, sigma=8, night=start))
    #tests.append(dict(name="T1_airmass_dec-40", ra=ra_mid, dec=-40, sigma=8, night=start))

    # T2 : transit 1.8 h après le début de la nuit => HA(début) ≈ -27°
    #tests.append(dict(name="T2_east_cut_dec-25",
    #                  ra=lst_deg(start + 1.8 * u.hour), dec=-25, sigma=10, night=start))

    #tests.append(dict(name="T3_pole_dec+88", ra=ra_mid, dec=+85, sigma=8, night=start))

    sun = sky_direction(get_sun(mid))
    #tests.append(dict(name="T4_sun_ra", ra=sun.ra.deg, dec=sun.dec.deg, sigma=15, night=start))

    # T5 : autre nuit avec la Lune haute (pleine lune le 24 nov 2026)
    start5, end5 = night_bounds("2026-11-22")
    mid5 = start5 + (end5 - start5) / 2
    moon_gcrs = get_body("moon", mid5, PALOMAR)
    moon = sky_direction(moon_gcrs)
    moon_alt = moon_gcrs.transform_to(AltAz(obstime=mid5, location=PALOMAR)).alt.deg
    print(f"[T5] altitude Lune à minuit : {moon_alt:.0f}° (doit être > 20°)")
    #tests.append(dict(name="T5_moon", ra=moon.ra.deg, dec=moon.dec.deg, sigma=15, night=start5))

    # T6 : point du plan galactique (b=0, Dec>-10) dont la RA est la plus proche du transit
    l = np.linspace(0, 360, 3601)
    g = SkyCoord(l=l * u.deg, b=0 * u.deg, frame="galactic").icrs
    ok = g.dec.deg > -10
    d = np.abs((g.ra.deg - ra_mid + 180) % 360 - 180)
    d[~ok] = np.inf
    i = np.argmin(d)
    #tests.append(dict(name="T6_galactic_plane", ra=g.ra.deg[i], dec=g.dec.deg[i],
    #                  sigma=15, night=start))
    print(f"[T6] l={l[i]:.1f}, b=0 -> RA={g.ra.deg[i]:.1f}, Dec={g.dec.deg[i]:.1f}")

    # T7: test la constraint AvoidSkyRegionConstraint avec une zone de 10° autour d'un point du ciel
    constraint = dict(type="AvoidSkyRegionConstraint", args=["340deg 3deg", "8 deg"])
    tests.append(dict(name="T7_avoid_sky_region", ra=340, dec=3, sigma=18, night=start, constraints=[constraint]))
    return tests


if __name__ == "__main__":
    for t in build_tests():
        print(f"{t['name']:<22} RA={t['ra']:7.2f}  Dec={t['dec']:+6.2f}  "
              f"sigma={t['sigma']}  début de nuit UTC={t['night'].iso[:16]}")
        generate_and_plot_fake_skymap(center_coord=SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg, frame="icrs"),
                                      sigma_deg=t["sigma"], out_file=f"{t['name']}.fits")

        # set up the m4opt schedule command
        constraints = [
            {"type": "AirmassConstraint", "args": ["2", "1"]},
            {"type": "GalacticLatitudeConstraint", "args": ["7 deg"]},
        ] 

        command = [
                    "m4opt",
                    "schedule",
                    f"{t['name']}.fits",
                    f"schedule_{t['name']}.ecsv",
                    f"--exptime-min=300s",
                    f"--bandpass=r",
                    f"--deadline=10h",
                    f"--time-step=1min",
                    f"--max-fields=50",
                    f"--timelimit=10min",
                    f"--jobs=8",
                    f"--event-time={t['night'].isot}",
                    "--mission=ztf",
                ]
        for c in constraints:
            command += ["--constraint", json.dumps(c)]
        if "constraints" in t:
            for c in t["constraints"]:
                command += ["--constraint", json.dumps(c)]
        print(f"\n=== Scenario={t['name']} ===", flush=True)
        _run(command)