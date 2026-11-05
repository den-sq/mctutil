# Imported reference calculations; only imports and execution/output boundaries are adapted.
from pathlib import Path


def main(output_dir="."):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	import numpy as np, matplotlib
	matplotlib.use("Agg"); import matplotlib.pyplot as plt
	from mctutil.phase.phase_tomo import (single_distance_multimaterial, retrieve_phase, fbp,
							forward_project, predict_filter_error,
							fit_material_line, wavelength_m)

	E, px, z = 20.0, 0.8e-6, 0.05
	lam = wavelength_m(E)
	nx, ny, n_ang = 512, 64, 181
	angles = np.linspace(0, 180, n_ang, endpoint=False)
	mus = np.array([60.8, 100.0, 140.0])
	dls = np.array([9.12e-8, 1.50e-7, 1.20e-7])      # mats 1,2 on s=1.5e-9 line; mat 3 43% off
	fit = fit_material_line(mus, dls, through_origin=True)
	print(f"LS line slope={fit['slope']:.3e}  eps={np.array2string(fit['eps'], precision=2)}")

	Rm, r2, r3 = 80e-6, 30e-6, 26e-6
	c2, c3 = (-32e-6, 8e-6), (30e-6, -14e-6)
	xs = (np.arange(nx)-nx/2+0.5)*px
	ch = lambda u, r: 2*np.sqrt(np.clip(r**2-u**2, 0, None))
	fy = np.fft.fftfreq(ny, px)[:, None]; fxg = np.fft.fftfreq(nx, px)[None, :]
	H = np.exp(-1j*np.pi*lam*z*(fxg**2+fy**2))
	data = np.empty((n_ang, ny, nx), np.float32)
	tj_true = np.empty((3, n_ang, nx))
	for a, th in enumerate(np.deg2rad(angles)):
		t2 = ch(xs-(c2[0]*np.cos(th)+c2[1]*np.sin(th)), r2)
		t3 = ch(xs-(c3[0]*np.cos(th)+c3[1]*np.sin(th)), r3)
		t1 = ch(xs, Rm)-t2-t3
		tj_true[:, a] = (t1, t2, t3)
		B = mus@tj_true[:, a]; ph = -(2*np.pi/lam)*(dls@tj_true[:, a])
		u0 = np.broadcast_to(np.exp(-B/2+1j*ph), (ny, nx))
		data[a] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2
	print(f"max|phi| = {np.abs(ph).max():.2f} rad")

	res = single_distance_multimaterial(data, E, px, z, angles, mus, dls,
										n_iter=1, rows=[ny//2])
	# oracle: correction from TRUE thicknesses
	s, eps = fit['slope'], fit['eps']
	psi_t = np.einsum('j,jax->ax', -(2*np.pi/lam)*eps, tj_true)[:, None, :]
	Bsyn_t = np.einsum('j,jax->ax', mus, tj_true)[:, None, :]
	r0 = retrieve_phase(data, E, px, z, method='single', slope=s, air_ratio=0.05)
	dB_t = predict_filter_error(psi_t, np.exp(-Bsyn_t), E, px, z, s)
	mu_or = fbp((r0['B'][:, ny//2, :]-dB_t[:, 0, :]), angles, px)

	Xg, Yg = np.meshgrid(xs, xs)
	regs = {'matrix': (np.hypot(Xg, Yg) < 0.75*Rm) & (np.hypot(Xg-c2[0], Yg-c2[1]) > 1.3*r2)
						& (np.hypot(Xg-c3[0], Yg-c3[1]) > 1.3*r3),
			'mat2': np.hypot(Xg-c2[0], Yg-c2[1]) < 0.7*r2,
			'mat3': np.hypot(Xg-c3[0], Yg-c3[1]) < 0.7*r3}
	rim = (np.hypot(Xg-c3[0], Yg-c3[1]) > 0.85*r3) & (np.hypot(Xg-c3[0], Yg-c3[1]) < 1.3*r3)
	mu_true = np.where(np.hypot(Xg, Yg) <= Rm, mus[0], 0.)
	for c, r, m in [(c2, r2, mus[1]), (c3, r3, mus[2])]:
		mu_true = np.where(np.hypot(Xg-c[0], Yg-c[1]) <= r, m, mu_true)
	lab_true = np.argmin(np.abs(mu_true[..., None]-np.concatenate([[0.], mus])), -1)

	stages = [(f"iter {i} " + ("(baseline)" if i == 0 else "corrected"), m)
				for i, m in enumerate(res['history_mu'])] + [("oracle correction", mu_or)]
	print(f"\n{'stage':28s}{'matrix':>9s}{'mat2':>9s}{'mat3':>9s}{'rim RMSE':>10s}{'misclass':>10s}")
	for name, mu_sl in stages:
		mu_sl = np.asarray(mu_sl)[0] if np.asarray(mu_sl).ndim == 3 else np.asarray(mu_sl)
		errs = [100*(mu_sl[regs[k]].mean()/m-1) for k, m in zip(regs, mus)]
		rr = np.sqrt(((mu_sl-mu_true)[rim]**2).mean())
		from mctutil.phase.phase_tomo import classify_materials
		mc = 100*np.mean(classify_materials(mu_sl, mus) != lab_true)
		print(f"{name:28s}{errs[0]:+8.2f}%{errs[1]:+8.2f}%{errs[2]:+8.2f}%"
				f"{rr:9.2f}{mc:9.2f}%")

	# delta volume from classification
	d_class = res['delta']
	d_true = np.where(np.hypot(Xg, Yg) <= Rm, dls[0], 0.)
	for c, r, dv in [(c2, r2, dls[1]), (c3, r3, dls[2])]:
		d_true = np.where(np.hypot(Xg-c[0], Yg-c[1]) <= r, dv, d_true)
	print(f"\nclassified-delta RMSE = {np.sqrt(((d_class-d_true)**2).mean()):.2e} "
			f"(delta range {dls.min():.1e}..{dls.max():.1e}); "
			f"exact on {100*np.mean(np.isclose(d_class, d_true)):.2f}% of voxels")

	b0 = np.asarray(res['history_mu'][0]); b0 = b0[0] if b0.ndim == 3 else b0
	b2 = np.asarray(res['history_mu'][-1]); b2 = b2[0] if b2.ndim == 3 else b2
	fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
	for A, ttl, axi in [(b0-mu_true, "baseline error", ax[0]),
						(b2-mu_true, "after damped iterations", ax[1])]:
		im = axi.imshow(A, extent=[xs[0]*1e6, xs[-1]*1e6]*2, cmap='RdBu_r',
						vmin=-40, vmax=40)
		axi.set_title(f"$\\mu$ {ttl} [1/m]"); axi.set_xlabel("x [$\\mu$m]")
		axi.set_xlim(-100, 100); axi.set_ylim(-100, 100); plt.colorbar(im, ax=axi)
	mid = np.argmin(np.abs(xs-c3[1]))
	row_y = np.argmin(np.abs(xs - c3[1]))
	ax[2].plot(xs*1e6, mu_true[row_y], 'k:', lw=1.2, label="truth")
	ax[2].plot(xs*1e6, b0[row_y], label="baseline")
	ax[2].plot(xs*1e6, b2[row_y], label="corrected (2 it.)")
	ax[2].plot(xs*1e6, mu_or[row_y], '--', label="oracle")
	ax[2].set_xlim(-20, 80); ax[2].set_xlabel("x [$\\mu$m]"); ax[2].set_ylabel("$\\mu$ [1/m]")
	ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("profile through mat 3")
	fig.tight_layout(); fig.savefig(output_dir / "multimat_single.png", dpi=150)
	print("figure saved.")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run the supplied synthetic phase validation.")
	parser.add_argument("output_dir", nargs="?", default=".")
	main(parser.parse_args().output_dir)
