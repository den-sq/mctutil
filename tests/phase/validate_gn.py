"""Gauss-Newton validation: (A) projection-domain accuracy vs the linear
defect-corrected path at two field strengths on the nano phantom;
(B) affine material line (alpha != 0), which the linear CTF cannot model."""

# Imported reference calculations; only imports and execution/output boundaries are adapted.
def main():
	import numpy as np
	from mctutil.phase.phase_tomo import retrieve_phase, gauss_newton_refine, wavelength_m

	E, lam = 20.0, wavelength_m(20.0); kp = 2*np.pi/lam

	def nano(scale):
		nx, ny, n_ang = 512, 64, 181
		px = 50e-9
		s = 1.5e-9; mus = scale*np.array([60.8, 100., 140.]); dls = s*mus
		Rm, r2, r3 = 9e-6, 3.2e-6, 2.6e-6
		c2, c3 = (-3.5e-6, 1e-6), (3.4e-6, -1.6e-6)
		xs = (np.arange(nx)-nx/2+0.5)*px
		ch = lambda u, r: 2*np.sqrt(np.clip(r**2-u**2, 0, None))
		fy = np.fft.fftfreq(ny, px)[:, None]; fx = np.fft.fftfreq(nx, px)[None, :]
		zs = [0.004, 0.010, 0.025]
		Hs = [np.exp(-1j*np.pi*lam*z*(fx**2+fy**2)) for z in zs]
		dat = np.empty((3, n_ang, ny, nx), np.float32)
		Bt = np.empty((n_ang, ny, nx)); pt = np.empty((n_ang, ny, nx))
		for a, th in enumerate(np.deg2rad(np.linspace(0, 180, n_ang, False))):
			t2 = ch(xs-(c2[0]*np.cos(th)+c2[1]*np.sin(th)), r2)
			t3 = ch(xs-(c3[0]*np.cos(th)+c3[1]*np.sin(th)), r3)
			t1 = ch(xs, Rm)-t2-t3
			B = mus@np.stack([t1, t2, t3]); ph = -kp*(dls@np.stack([t1, t2, t3]))
			Bt[a] = B; pt[a] = ph
			u0 = np.broadcast_to(np.exp(-B/2+1j*ph), (ny, nx))
			for d, H in enumerate(Hs):
				dat[d, a] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2
		return dat, zs, px, Bt, pt, s

	def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

	for scale in (1.0, 3.0):
		dat, zs, px, Bt, pt, s = nano(scale)
		print(f"--- nano phantom x{scale:.0f}: max|phi| = {np.abs(pt).max():.2f} rad ---")
		pure = retrieve_phase(dat, E, px, zs, method='ctf', pad=False,
								reg_phi=(1e-10, 1e-12), reg_B=1e-12)
		dc = retrieve_phase(dat, E, px, zs, method='ctf', slope=s, pad=False,
							reg_phi=(1e-10, 1e-12), reg_B=1e-12)
		gn = gauss_newton_refine(dat, E, px, zs, dc['B'], dc['phi'],
									n_newton=2, n_cg=12, slope=s, pad=False)
		for tag, r in (("pure CTF", pure), ("CTF+defect corr.", dc),
						("GN refined", gn)):
			print(f"  {tag:18s} B RMSE {rmse(r['B'], Bt):.3e}   "
					f"phi RMSE {rmse(r['phi'], pt):.3e}")
		print(f"  nonlinear data residual: {gn['residuals'][:,0].mean():.3e} -> "
				f"{gn['residuals'][:,-1].mean():.3e}")

	# --- (B) affine line: delta = alpha + s*mu, envelope known ---
	nx, ny, n_ang = 384, 8, 8
	px = 50e-9
	alpha, s = 2e-7, 1.5e-9
	mus = np.array([800., 1600.]); dls = alpha + s*mus
	Rm, r2c, r2r = 7e-6, 2e-6, 2.5e-6
	xs = (np.arange(nx)-nx/2+0.5)*px
	ch = lambda u, r: 2*np.sqrt(np.clip(r**2-u**2, 0, None))
	fy = np.fft.fftfreq(ny, px)[:, None]; fx = np.fft.fftfreq(nx, px)[None, :]
	zs = [0.004, 0.010, 0.025]
	Hs = [np.exp(-1j*np.pi*lam*z*(fx**2+fy**2)) for z in zs]
	dat = np.empty((3, n_ang, ny, nx), np.float32)
	Bt = np.empty((n_ang, ny, nx)); pt = np.empty((n_ang, ny, nx))
	Tt = np.empty((n_ang, ny, nx))
	for a, th in enumerate(np.deg2rad(np.linspace(0, 180, n_ang, False))):
		t2 = ch(xs-r2c*np.cos(th), r2r); t1 = ch(xs, Rm)-t2
		T = t1+t2; B = mus[0]*t1+mus[1]*t2
		ph = -kp*(dls[0]*t1+dls[1]*t2)
		Bt[a] = B; pt[a] = ph; Tt[a] = T
		u0 = np.broadcast_to(np.exp(-B/2+1j*ph), (ny, nx))
		for d, H in enumerate(Hs):
			dat[d, a] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2
	print(f"--- affine phantom: alpha={alpha:.1e}, max|phi| = {np.abs(pt).max():.2f} rad ---")
	init = retrieve_phase(dat[0], E, px, zs[0], method='single', slope=s,
							alpha=alpha, envelope=Tt, pad=False)
	gn_a = gauss_newton_refine(dat, E, px, zs, init['B'], init['phi'],
								n_newton=6, n_cg=40, slope=s, alpha=alpha,
								envelope=Tt, pad=False)
	s_wrong = float(np.mean(dls/mus))          # best through-origin surrogate
	init_w = retrieve_phase(dat[0], E, px, zs[0], method='single', slope=s_wrong,
							pad=False)
	gn_w = gauss_newton_refine(dat, E, px, zs, init_w['B'], init_w['phi'],
								n_newton=6, n_cg=40, slope=s_wrong, pad=False)
	for tag, r in (("affine Paganin init", init), ("GN affine model", gn_a),
					("GN wrong (origin) model", gn_w)):
		print(f"  {tag:24s} B RMSE {rmse(r['B'], Bt):.3e}   "
				f"phi RMSE {rmse(r['phi'], pt):.3e}")
	print(f"  GN affine residual: {gn_a['residuals'][:,0].mean():.3e} -> "
			f"{gn_a['residuals'][:,-1].mean():.3e}")


if __name__ == "__main__":
	main()
