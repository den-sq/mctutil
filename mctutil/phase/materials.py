"""Build the (mu, delta) material table for phase_tomo from composition,
density, and energy, using xraylib (delta from f1/electron density; mu from
total cross sections, i.e. photoelectric + Compton + Rayleigh -- do NOT use
Henke f2 alone for mu at hard X-ray energies).
Inputs per material: chemical formula OR elemental mass fractions,
measured bulk density of the actual stock [g/cm3], energy [keV]."""
import numpy as np, xraylib
from .phase_tomo import fit_material_line

def material_point(formula, density, energy_kev):
	delta = 1.0 - xraylib.Refractive_Index_Re(formula, energy_kev, density)
	mu = xraylib.CS_Total_CP(formula, energy_kev) * density * 100.0   # 1/m
	return mu, delta

def build_table(materials, energy_kev, weights=None, through_origin=True):
	"""materials: dict name -> (formula, density). Returns dict with arrays
	mu, delta (module-ready), the LS line fit, and the per-material
	single-filter artifact budget (plateau dB per metre of feature)."""
	names = list(materials)
	mu, dl = np.array([material_point(*materials[n], energy_kev)
						for n in names]).T
	fit = fit_material_line(mu, dl, weights, through_origin=through_origin)
	return dict(names=names, mu=mu, delta=dl, fit=fit,
				plateau_dB_per_m=np.abs(fit['eps'])/fit['slope'])

if __name__ == "__main__":
	mats = {'PMMA': ('C5H8O2', 1.19), 'POM': ('CH2O', 1.41),
			'PS': ('C8H8', 1.06), 'PTFE': ('C2F4', 2.20)}
	for E in [15., 20., 25., 30., 40.]:
		t = build_table(mats, E)
		print(f"E={E:4.0f} keV  s={t['fit']['slope']:.3e}  "
				f"rms_eps={t['fit']['rms_eps']:.2e}")
	t = build_table(mats, 20.0)
	for n, m, d, e in zip(t['names'], t['mu'], t['delta'], t['fit']['eps']):
		print(f"{n:5s} mu={m:7.1f} /m  delta={d:.3e}  eps={e:+.2e}")
