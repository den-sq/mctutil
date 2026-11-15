# Imported reference calculations; only imports and execution/output boundaries are adapted.
from pathlib import Path


def main(output_dir="."):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	import numpy as np, matplotlib
	matplotlib.use("Agg"); import matplotlib.pyplot as plt
	from mctutil.phase.phase_tomo import retrieve_phase, fbp, wavelength_m

	E = 20.0; lam = wavelength_m(E)
	n_ang = 181; angles = np.linspace(0, 180, n_ang, endpoint=False)
	s_true = 1.5e-9
	mus = np.array([60.8, 100.0, 140.0]); dls = s_true*mus   # collinear via origin

	def make_dataset(px, nx, ny, Rm, r2, r3, c2, c3, zs):
		xs = (np.arange(nx)-nx/2+0.5)*px
		fy = np.fft.fftfreq(ny, px)[:,None]; fx = np.fft.fftfreq(nx, px)[None,:]
		Hs = [np.exp(-1j*np.pi*lam*z*(fx**2+fy**2)) for z in zs]
		chord = lambda u,r: 2*np.sqrt(np.clip(r**2-u**2,0,None))
		data = np.empty((len(zs), n_ang, ny, nx), np.float32)
		for a, th in enumerate(np.deg2rad(angles)):
			t2 = chord(xs-(c2[0]*np.cos(th)+c2[1]*np.sin(th)), r2)
			t3 = chord(xs-(c3[0]*np.cos(th)+c3[1]*np.sin(th)), r3)
			t1 = chord(xs, Rm)-t2-t3
			B = mus[0]*t1+mus[1]*t2+mus[2]*t3
			ph = -(2*np.pi/lam)*(dls[0]*t1+dls[1]*t2+dls[2]*t3)
			u0 = np.broadcast_to(np.exp(-B/2+1j*ph), (ny, nx))
			for d, H in enumerate(Hs):
				data[d, a] = np.abs(np.fft.ifft2(np.fft.fft2(u0)*H))**2
		return data, xs, np.abs(ph).max()

	def region_table(delta_map, xs, Rm, r2, r3, c2, c3, label):
		Xg, Yg = np.meshgrid(xs, xs)
		regs = {'matrix': (np.hypot(Xg,Yg)<0.75*Rm) & (np.hypot(Xg-c2[0],Yg-c2[1])>1.3*r2)
							& (np.hypot(Xg-c3[0],Yg-c3[1])>1.3*r3),
				'mat2': np.hypot(Xg-c2[0],Yg-c2[1])<0.7*r2,
				'mat3': np.hypot(Xg-c3[0],Yg-c3[1])<0.7*r3}
		errs = {}
		for i,(k,m) in enumerate(regs.items()):
			errs[k] = 100*(delta_map[m].mean()/dls[i]-1)
			print(f"  {label:22s}{k:8s} true={dls[i]:.3e}  rec={delta_map[m].mean():.3e}  ({errs[k]:+.2f}%)")
		return errs

	# ===== A: microCT scale (0.8 um px, strong phase): 'single' & 'ctf_homogeneous'
	pxA, nxA, nyA = 0.8e-6, 512, 64
	geomA = dict(Rm=80e-6, r2=30e-6, r3=26e-6, c2=(-32e-6,8e-6), c3=(30e-6,-14e-6))
	zsA = [0.02, 0.05, 0.10]
	dataA, xsA, phmaxA = make_dataset(pxA, nxA, nyA, **geomA, zs=zsA)
	print(f"A: micro scale, max|phi| = {phmaxA:.2f} rad (strong-phase / TIE regime)")
	sngA = retrieve_phase(dataA[0], E, pxA, zsA[0], method='single', slope=s_true, air_ratio=0.05)
	homA = retrieve_phase(dataA, E, pxA, zsA, method='ctf_homogeneous', slope=s_true, air_ratio=0.05)
	row = nyA//2
	d_sng = fbp(sngA['delta_proj'][:,row,:], angles, pxA)
	d_hom = fbp(homA['delta_proj'][:,row,:], angles, pxA)
	eS = region_table(d_sng, xsA, *geomA.values(), label="single (z=2cm)")
	eH = region_table(d_hom, xsA, *geomA.values(), label="ctf_homogeneous (3z)")

	# ===== B: nano-holotomography scale (50 nm px, weak object): full 'ctf'
	pxB, nxB, nyB = 50e-9, 512, 64
	geomB = dict(Rm=9e-6, r2=3.2e-6, r3=2.6e-6, c2=(-3.5e-6,1e-6), c3=(3.4e-6,-1.6e-6))
	zsB = [0.004, 0.010, 0.025]
	dataB, xsB, phmaxB = make_dataset(pxB, nxB, nyB, **geomB, zs=zsB)
	print(f"B: nano scale, max|phi| = {phmaxB:.3f} rad (weak-object / CTF regime)")
	ctfP = retrieve_phase(dataB, E, pxB, zsB, method='ctf', pad=False,
							reg_phi=(1e-10, 1e-12), reg_B=1e-12, air_ratio=0.05)
	d_pure = fbp(ctfP['delta_proj'][:,nyB//2,:], angles, pxB)
	eP = region_table(d_pure, xsB, *geomB.values(), label="ctf pure (no prior)")
	ctfB = retrieve_phase(dataB, E, pxB, zsB, method='ctf', slope=s_true, pad=False,
							reg_phi=(1e-10, 1e-12), reg_B=1e-12, air_ratio=0.05)
	d_ctf = fbp(ctfB['delta_proj'][:,nyB//2,:], angles, pxB)
	eC = region_table(d_ctf, xsB, *geomB.values(), label="ctf+GN (lf prior)")

	# ===== figure =====
	fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
	im=ax[0].imshow(d_sng*1e7, extent=[xsA[0]*1e6,xsA[-1]*1e6]*2, cmap='viridis', vmin=0, vmax=2.4)
	ax[0].set_title("micro: single-distance $\\delta$"); ax[0].set_xlim(-100,100); ax[0].set_ylim(-100,100)
	plt.colorbar(im, ax=ax[0], label="$\\delta\\times10^{7}$"); ax[0].set_xlabel("x [$\\mu$m]")
	im=ax[1].imshow(d_ctf*1e7, extent=[xsB[0]*1e6,xsB[-1]*1e6]*2, cmap='viridis', vmin=0, vmax=2.4)
	ax[1].set_title("nano: full-CTF $\\delta$ (no material priors)"); ax[1].set_xlim(-11,11); ax[1].set_ylim(-11,11)
	plt.colorbar(im, ax=ax[1], label="$\\delta\\times10^{7}$"); ax[1].set_xlabel("x [$\\mu$m]")
	mid=nxA//2
	Xg,Yg=np.meshgrid(xsA,xsA)
	dtr=np.where(np.hypot(Xg,Yg)<=geomA['Rm'], dls[0], 0)
	for c,r,dv in [(geomA['c2'],geomA['r2'],dls[1]),(geomA['c3'],geomA['r3'],dls[2])]:
		dtr=np.where(np.hypot(Xg-c[0],Yg-c[1])<=r, dv, dtr)
	ax[2].plot(xsA*1e6, d_sng[mid]*1e7, label="single-distance")
	ax[2].plot(xsA*1e6, d_hom[mid]*1e7, '--', label="ctf_homogeneous")
	ax[2].plot(xsA*1e6, dtr[mid]*1e7, 'k:', lw=1.2, label="truth")
	ax[2].set_xlim(-100,100); ax[2].set_xlabel("x [$\\mu$m]"); ax[2].set_ylabel("$\\delta\\times10^{7}$")
	ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("micro-scale line profile")
	fig.tight_layout(); fig.savefig(output_dir / "tomo_validation.png", dpi=150)
	print("figure saved.")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run the supplied synthetic phase validation.")
	parser.add_argument("output_dir", nargs="?", default=".")
	main(parser.parse_args().output_dir)
