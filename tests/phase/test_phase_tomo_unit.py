"""Assertion-based unit tests for phase_tomo, encoding the review's
reproductions plus core-math checks. numpy+scipy only; fast (<1 min)."""

# Imported reference calculations; only imports and execution/output boundaries are adapted.
def test_reference_unit_suite():
	import numpy as np
	from mctutil.phase.phase_tomo import (retrieve_phase, fbp, forward_project,
							segment_materials, extract_material_points,
							estimate_line_slope, single_distance_multimaterial,
							gauss_newton_refine, _fresnel_ops, _fresnel_jvp,
							_fresnel_vjp, wavelength_m)

	def expect_raise(fn, exc, tag):
		try:
			fn()
		except exc:
			print(f"PASS {tag}"); return
		raise AssertionError(f"{tag}: expected {exc.__name__}")

	E, lam = 20.0, wavelength_m(20.0)
	rng = np.random.default_rng(0)

	# ---- shape contracts (CORE-01) ----
	I2 = np.full((16, 32), np.exp(-0.01), np.float64)
	r = retrieve_phase(I2, E, 1e-6, 0.01, method='single', slope=1e-9, dtype=np.float64)
	assert r['B'].shape == (16, 32) and r['phi'].shape == (16, 32)
	r = retrieve_phase(np.stack([I2]*3), E, 1e-6, 0.01, method='single', slope=1e-9)
	assert r['B'].shape == (3, 16, 32)
	r = retrieve_phase(np.ones((5, 1, 32), np.float32), E, 1e-6, 0.01,
						method='single', slope=1e-9)
	assert r['B'].shape == (5, 1, 32), "ny=1 axis must be preserved"
	fbp(r['B'][:, 0, :], np.linspace(0, 180, 5, endpoint=False), 1e-6)  # no crash
	r = retrieve_phase(np.ones((2, 4, 8, 32), np.float32), E, 1e-6, [0.01, 0.02],
						method='ctf_homogeneous', slope=1e-9)
	assert r['B'].shape == (4, 8, 32)
	r = retrieve_phase(np.ones((2, 8, 32), np.float32), E, 1e-6, [0.01, 0.02],
						method='ctf_homogeneous', slope=1e-9)
	assert r['B'].shape == (8, 32)
	print("PASS shape contracts (incl. ny=1 preservation)")

	# ---- uniform-field identity + chunk invariance + dtype ----
	B0 = 0.017
	I = np.full((7, 12, 48), np.exp(-B0))
	r64 = retrieve_phase(I, E, 1e-6, 0.02, method='single', slope=3e-9,
							dtype=np.float64)
	assert np.allclose(r64['B'], B0, atol=1e-8), "uniform field identity"
	ra = retrieve_phase(I, E, 1e-6, 0.02, method='single', slope=3e-9, chunk=3)
	rb = retrieve_phase(I, E, 1e-6, 0.02, method='single', slope=3e-9, chunk=64)
	assert np.array_equal(ra['B'], rb['B']), "chunk invariance"
	print("PASS uniform-field identity, chunk invariance, float64 path")

	# ---- CORE-02: per-projection envelope across chunk boundary ----
	n_ang, ny, nx = 33, 8, 64
	env = np.full((n_ang, ny, nx), 3e-4, np.float64)
	Ienv = np.full((n_ang, ny, nx), np.exp(-0.01), np.float32)
	r = retrieve_phase(Ienv, E, 1e-6, 0.01, method='single', slope=1e-9,
						alpha=2e-8, envelope=env, chunk=32)
	assert r['B'].shape == (n_ang, ny, nx) and np.all(np.isfinite(r['phi']))
	print("PASS envelope chunking (33 angles, chunk=32)")
	expect_raise(lambda: retrieve_phase(Ienv, E, 1e-6, 0.01, method='single',
										slope=1e-9, alpha=2e-8),
					ValueError, "alpha without envelope raises (CORE-04)")

	# ---- CORE-03: CTF rejects affine inputs ----
	expect_raise(lambda: retrieve_phase(np.ones((2, 4, 8, 32), np.float32), E,
										1e-6, [0.01, 0.02], method='ctf_homogeneous',
										slope=1e-9, alpha=1e-8),
					NotImplementedError, "CTF alpha rejection (CORE-03)")

	# ---- MM-01: projector discards out-of-detector mass; adjointness ----
	img = np.zeros((8, 8)); img[7, 7] = 1.0   # corner (x=+3.5,y=+3.5)px -> u=4.95 at 45deg
	s45 = forward_project(img, [45.0], 1e-6)
	assert s45.sum() == 0.0, "corner at 45 deg must be discarded, not clamped"
	disk = (np.hypot(*np.meshgrid(np.arange(24)-11.5, np.arange(24)-11.5)) < 8)
	sd = forward_project(disk.astype(float), [0., 30., 77.], 1e-6)
	assert np.allclose(sd.sum(1), disk.sum()*1e-6, rtol=1e-12), "mass conservation"
	x = rng.normal(size=(16, 16)); y = rng.normal(size=(5, 16))
	angles5 = [0., 22., 45., 90., 133.]
	Ax = forward_project(x, angles5, 1e-6)
	xs = (np.arange(16)-8+0.5)*1e-6
	Xg, Yg = np.meshgrid(xs, xs)
	Aty = np.zeros(16*16)
	for a, th in enumerate(np.deg2rad(angles5)):
		q = ((Xg*np.cos(th)+Yg*np.sin(th)).ravel() - xs[0]) / 1e-6
		i0 = np.floor(q).astype(int); w = q - i0
		m0 = (i0 >= 0) & (i0 < 16); m1 = (i0 >= -1) & (i0 < 15)
		Aty[m0] += y[a][i0[m0]]*(1-w[m0])
		Aty[m1] += y[a][i0[m1]+1]*w[m1]
	Aty = Aty.reshape(16, 16)
	assert abs(np.sum(Ax*y) - np.sum(x*Aty)*1e-6) < 1e-9*abs(np.sum(Ax*y)), \
		"forward/backprojector adjointness"
	print("PASS projector bounds, mass conservation, adjointness (MM-01)")

	# ---- fbp validation ----
	expect_raise(lambda: fbp(np.ones((5, 32)), [0., 90.], 1e-6),
					ValueError, "fbp angle-count validation")

	# ---- SEG-01 / table validation ----
	mmap = np.zeros((64, 64)); mmap[10:30, 10:30] = 50.0
	expect_raise(lambda: segment_materials(mmap, [50.0, 120.0]),
					ValueError, "missing class core raises (SEG-01)")
	expect_raise(lambda: segment_materials(mmap, [50.0, 50.0]),
					ValueError, "duplicate mu_table raises")

	# ---- CLUSTER-01 ----
	cmap = np.full((64, 64), 7.0)
	expect_raise(lambda: extract_material_points(cmap, cmap*1e-9, 2),
					ValueError, "cluster shortfall raises (CLUSTER-01)")
	g1 = rng.normal(10, 0.3, 4000); g2 = rng.normal(60, 0.3, 4000)
	mu_t = np.concatenate([g1, g2]); dl_t = mu_t*2e-9 + rng.normal(0, 2e-10, 8000)
	pts = extract_material_points(mu_t.reshape(80, 100), dl_t.reshape(80, 100), 2)
	assert np.allclose(np.sort(pts['centroids'][:, 0]), [10, 60], atol=0.5)
	print("PASS cluster extraction happy path")

	# ---- SLOPE-01: guards + recovery ----
	nxs, nys = 192, 8
	xs2 = (np.arange(nxs)-nxs/2+0.5)*0.5e-6
	s_true = 2e-9
	Bp = 0.0006*(np.exp(-((xs2+15e-6)/1.5e-6)**2) + np.exp(-((xs2-12e-6)/1.0e-6)**2))
	php = -(2*np.pi/lam)*s_true*Bp
	fy = np.fft.fftfreq(nys, 0.5e-6)[:, None]; fx = np.fft.fftfreq(nxs, 0.5e-6)[None, :]
	zsl = [0.005, 0.012, 0.02]   # 3 distances: the intended-use configuration
	dat = np.empty((3, 1, nys, nxs), np.float32)
	u0 = np.broadcast_to(np.exp(-Bp/2+1j*php), (nys, nxs))
	for d, z in enumerate(zsl):
		H = np.exp(-1j*np.pi*lam*z*(fx**2+fy**2))
		dat[d, 0] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2
	s_hat, info = estimate_line_slope(dat, E, 0.5e-6, zsl, angle_step=1, pad=False)
	assert abs(s_hat/s_true - 1) < 0.07, f"slope recovery: {s_hat:.3e}"
	assert info['explained_fraction'] > 0.95 and info['n_bins'] >= 32
	expect_raise(lambda: estimate_line_slope(dat, E, 0.5e-6, zsl, angle_step=1,
												pad=False, chi_band=(50., 60.)),
					ValueError, "empty chi_band raises (SLOPE-01)")
	expect_raise(lambda: estimate_line_slope(dat, E, 0.5e-6, zsl, angle_step=1,
												pad=False, s_range=(1e-11, 1e-10)),
					ValueError, "boundary minimum raises (unbracketed)")
	print(f"PASS slope estimator (s_hat err {100*(s_hat/s_true-1):+.2f}%, "
			f"explained {info['explained_fraction']:.3f})")

	# ---- multimaterial smoke at ny=1 (the (n_ang,1,nx) failure class) ----
	nxm, n_angm = 96, 24
	angm = np.linspace(0, 180, n_angm, endpoint=False)
	mus_m = np.array([50.0, 120.0]); s_m = 2e-9; dls_m = s_m*mus_m
	xsm = (np.arange(nxm)-nxm/2+0.5)*1e-6
	chd = lambda u, r: 2*np.sqrt(np.clip(r**2-u**2, 0, None))
	fym = np.fft.fftfreq(1, 1e-6)[:, None]; fxm = np.fft.fftfreq(nxm, 1e-6)[None, :]
	Hm = np.exp(-1j*np.pi*lam*0.01*(fxm**2+fym**2))
	dm = np.empty((n_angm, 1, nxm), np.float32)
	for a, th in enumerate(np.deg2rad(angm)):
		t2 = chd(xsm-(8e-6*np.cos(th)-5e-6*np.sin(th)), 10e-6)
		t1 = chd(xsm, 30e-6) - t2
		Bm = mus_m[0]*t1 + mus_m[1]*t2
		pm = -(2*np.pi/lam)*(dls_m[0]*t1 + dls_m[1]*t2)
		dm[a] = np.abs(np.fft.ifft2(np.fft.fft2(
			np.exp(-Bm/2+1j*pm)[None, :])*Hm))**2
	res = single_distance_multimaterial(dm, E, 1e-6, 0.01, angm, mus_m, dls_m,
										n_iter=1, rows=[0])
	assert res['mu'].shape == (1, nxm, nxm) and res['labels'].shape == (1, nxm, nxm)
	assert len(res['correction_norms']) == 1
	Xm, Ym = np.meshgrid(xsm, xsm)
	core = res['mu'][0][(np.hypot(Xm, Ym) < 24e-6)
						& (np.hypot(Xm-8e-6, Ym+5e-6) > 14e-6)]
	assert abs(core.mean()/mus_m[0] - 1) < 0.08
	print("PASS multimaterial pipeline on (n_ang, 1, nx) stack")

	# ---- Gauss-Newton: exact adjoint, residual decrease, contracts ----
	Bf = 0.02*rng.random((16, 32)); pf = 0.10*rng.random((16, 32))
	fyt = np.fft.fftfreq(16, 1e-6)[:, None]; fxt = np.fft.fftfreq(32, 1e-6)[None, :]
	Hs2 = [np.exp(-1j*np.pi*lam*z*(fyt**2 + fxt**2)) for z in (0.01, 0.03)]
	u0t, uzt = _fresnel_ops(Bf, pf, Hs2)
	dBt = rng.normal(size=(16, 32)); dpt = rng.normal(size=(16, 32))
	rrt = [rng.normal(size=(16, 32)) for _ in Hs2]
	dI = _fresnel_jvp(u0t, uzt, Hs2, dBt, dpt)
	gBt, gpt = _fresnel_vjp(u0t, uzt, Hs2, rrt)
	lhs = sum(np.sum(a*b) for a, b in zip(dI, rrt))
	rhs = np.sum(dBt*gBt) + np.sum(dpt*gpt)
	assert abs(lhs - rhs) < 1e-10*max(abs(lhs), 1.0), "JVP/VJP adjoint identity"
	init = retrieve_phase(dat, E, 0.5e-6, zsl, method='ctf', slope=s_true,
							pad=False, reg_phi=(1e-10, 1e-12), reg_B=1e-12)
	gn = gauss_newton_refine(dat, E, 0.5e-6, zsl, init['B'], init['phi'],
								n_newton=2, n_cg=10, slope=s_true, pad=False)
	resh = gn['residuals'][0]
	assert gn['B'].shape == init['B'].shape and len(resh) == 3
	assert resh[-1] < 0.3*resh[0], f"GN residual: {resh}"
	expect_raise(lambda: gauss_newton_refine(dat, E, 0.5e-6, zsl, init['B'],
												init['phi'], alpha=1e-8, pad=False),
					ValueError, "GN alpha without envelope raises")
	print(f"PASS Gauss-Newton adjoint + residual drop x{resh[0]/resh[-1]:.0f}")

	print("\nALL TESTS PASSED")


if __name__ == "__main__":
	test_reference_unit_suite()
