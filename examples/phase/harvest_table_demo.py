"""End-to-end demo: multi-distance scan -> harvest_material_table -> transfer.
Synthetic nano-holotomography scan of a 3-material collinear phantom."""

# Imported reference calculations; only imports and execution/output boundaries are adapted.
def main():
	import numpy as np
	from mctutil.phase.phase_tomo import harvest_material_table, wavelength_m  # numpy+scipy only

	E, px = 20.0, 50e-9
	lam = wavelength_m(E); nx, ny, n_ang = 512, 64, 181
	angles = np.linspace(0, 180, n_ang, endpoint=False)
	s_true = 1.5e-9; mus = np.array([60.8, 100., 140.]); dls = s_true*mus
	Rm, r2, r3 = 9e-6, 3.2e-6, 2.6e-6; c2 = (-3.5e-6, 1e-6); c3 = (3.4e-6, -1.6e-6)
	xs = (np.arange(nx)-nx/2+0.5)*px
	ch = lambda u, r: 2*np.sqrt(np.clip(r**2-u**2, 0, None))
	fy = np.fft.fftfreq(ny, px)[:, None]; fxg = np.fft.fftfreq(nx, px)[None, :]
	zs = [0.004, 0.010, 0.025]
	Hs = [np.exp(-1j*np.pi*lam*z*(fxg**2+fy**2)) for z in zs]
	data = np.empty((len(zs), n_ang, ny, nx), np.float32)
	for a, th in enumerate(np.deg2rad(angles)):
		t2 = ch(xs-(c2[0]*np.cos(th)+c2[1]*np.sin(th)), r2)
		t3 = ch(xs-(c3[0]*np.cos(th)+c3[1]*np.sin(th)), r3)
		t1 = ch(xs, Rm)-t2-t3
		B = mus@np.stack([t1, t2, t3])
		ph = -(2*np.pi/lam)*(dls@np.stack([t1, t2, t3]))
		u0 = np.broadcast_to(np.exp(-B/2+1j*ph), (ny, nx))
		for d, H in enumerate(Hs):
			data[d, a] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2

	tab = harvest_material_table(data, E, px, zs, angles, n_materials=3,
									rows=[ny//2], pad=False, slope_sensitivity=0.1, n_folds=4,
									reg_phi=(1e-10, 1e-12), reg_B=1e-12)
	print(f"bootstrap slope used: {tab['slope_used']:.4e} (true {s_true:.4e})")
	print(f"air point (additive bias): mu={tab['air_point'][0]:+.2f}, "
			f"delta={tab['air_point'][1]:+.2e}")
	print(f"{'':6s}{'mu [1/m]':>10s}{'err':>8s}{'delta':>12s}{'err':>8s}{'frac':>7s}")
	for m, d, fr, mt, dt in zip(tab['mu'], tab['delta'], tab['fractions'], mus, dls):
		print(f"  mat {m:10.1f}{100*(m/mt-1):+7.2f}%{d:12.3e}{100*(d/dt-1):+7.2f}%{fr:7.3f}")
	print(f"collinearity_residual (data-driven, mid-f) = "
			f"{tab['collinearity_residual']:.4f}")
	print(f"delta prior sensitivity d(delta)/d(ln s): "
			f"{np.array2string(tab['delta_slope_sensitivity'], precision=2)}")
	u = tab['uncertainty']
	print(f"angle-fold replicates: {u['n_folds_used']}/{u['n_folds']} usable; "
			f"slope fold spread = {u['slope_folds'].std(ddof=1):.2e}")
	for j in range(3):
		sg = np.sqrt(np.diag(u['centroid_cov_full_est'][j]))
		print(f"  mat{j+1} statistical (full-data est.): "
				f"mu +/- {sg[0]:.2f}, delta +/- {sg[1]:.2e}")
	print(f"line fit: s = {tab['line_fit']['slope']:.4e} "
			f"({100*(tab['line_fit']['slope']/s_true-1):+.2f}%), "
			f"rms_eps = {tab['line_fit']['rms_eps']:.2e}")
	try:
		from mctutil.phase.phase_tomo import transfer_table
		mu24, d24, info = transfer_table(tab['mu'], tab['delta'], E, 24.0)
		print(f"transferred to 24 keV: mu = {np.array2string(mu24, precision=1)}"
				f"  (basis cond {info['basis_condition']:.1f})")
	except ImportError:
		print("transfer step skipped: optional dependency xraylib not installed")


if __name__ == "__main__":
	main()
