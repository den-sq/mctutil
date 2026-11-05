"""
phase_tomo.py -- Phase retrieval for propagation-based X-ray tomography.

Two retrieval paths, per-projection (apply BEFORE tomographic reconstruction):

	method='single'   Single-distance generalized (affine) Paganin filter.
					Exact (within TIE) when the materials are collinear
					THROUGH THE ORIGIN (alpha=0): delta = s*mu. For
					alpha != 0 the envelope term is handled by a ONE-STEP
					defect correction (approximate; envelope T required).
					Classic Paganin: alpha=0, s = delta/mu of the material.
					Beltran two-material: s = (d2-d1)/(mu2-mu1).
					Returns B (=> mu volume after FBP); phi if alpha==0 or an
					envelope T is supplied (phi = -(2pi/lam)(alpha*T + s*B)).

	method='ctf'      Multi-distance contrast-transfer-function inversion:
					per-frequency Tikhonov least squares for (phi, B) jointly.
					VALIDITY: the linear phase signal is 2*sin(chi)*phi_tilde;
					for an extended object with phase scale phi0, quadratic
					terms dominate wherever 2*sin(chi) < phi0 -- so low-f phi
					is unrecoverable without a prior, at any distance set.
					If `slope` is given: (i) one nonlinear defect-correction
					step (Born-type residual subtraction) removes
					the quadratic model mismatch around the material-line
					prior, and (ii) low frequencies are anchored to that
					prior (blend at chi_max = 0.3) while high frequencies
					stay assumption-free. Without `slope`, the result is
					quantitative only where 2*sin(chi) >> phase scale.

	method='ctf_homogeneous'
					Multi-distance, single unknown B with phi slaved to the
					material line (needs slope; alpha=0 assumed unless
					envelope given). Best-conditioned option for collinear
					samples with several distances.

Dependencies: numpy, scipy (required); xraylib (optional, only for
transfer_table and infer_effective_energy).

Conventions: u ~ exp(+ikz), n = 1 - delta + i*beta, mu = 4*pi*beta/lambda,
phi = -(2*pi/lambda) * Int delta dz  (<= 0),  B = Int mu dz,  I0 = exp(-B).
Projections must be flat/dark corrected (unit background), co-registered on a
common effective pixel grid. Cone beam: pass z_eff and pixel_eff from
fresnel_scaling(); for multi-distance cone-beam data, magnification-rescale to
a common grid first.
"""
import numpy as np
from scipy import fft as sfft
from scipy import ndimage

# ----------------------------------------------------------------------
def wavelength_m(energy_kev):
	return 1.23984193e-9 / energy_kev

def fresnel_scaling(R1, R2, pixel_det):
	"""Cone beam -> equivalent parallel geometry (Fresnel scaling theorem).
	Returns (z_eff, magnification, pixel_eff). Use pixel_eff as pixel_m and
	z_eff as distance_m."""
	M = (R1 + R2) / R1
	return R1 * R2 / (R1 + R2), M, pixel_det / M

def fit_material_line(mu, delta, weights=None, through_origin=False):
	"""Fit delta = alpha + s*mu over the material set (weights ~ expected
	interface prominence). Returns dict with alpha, s, per-material off-line
	residuals eps (the artifact drivers), and their weighted rms."""
	mu = np.asarray(mu, float); delta = np.asarray(delta, float)
	w = np.ones_like(mu) if weights is None else np.asarray(weights, float)
	if through_origin:
		s = np.sum(w * mu * delta) / np.sum(w * mu**2); alpha = 0.0
	else:
		W = np.sum(w); mub = np.sum(w*mu)/W; db = np.sum(w*delta)/W
		s = np.sum(w*(mu-mub)*(delta-db)) / np.sum(w*(mu-mub)**2)
		alpha = db - s*mub
	eps = delta - (alpha + s*mu)
	return dict(alpha=alpha, slope=s, eps=eps,
				rms_eps=float(np.sqrt(np.sum(w*eps**2)/np.sum(w))))

# ----------------------------------------------------------------------
def _pad_setup(ny, nx, pad):
	py = sfft.next_fast_len(int(ny*1.5)) if pad else ny
	px_ = sfft.next_fast_len(int(nx*1.5)) if pad else nx
	y0, x0 = (py-ny)//2, (px_-nx)//2
	return py, px_, y0, x0

def _pad(a, py, px_, y0, x0):
	ny, nx = a.shape[-2:]
	return np.pad(a, [(0,0)]*(a.ndim-2)+[(y0,py-ny-y0),(x0,px_-nx-x0)], mode='edge')

def _f2_grid(py, px_, pixel):
	fy = sfft.fftfreq(py, pixel)[:, None]
	fx = sfft.rfftfreq(px_, pixel)[None, :]
	return fy**2 + fx**2

# ----------------------------------------------------------------------
def retrieve_phase(projections, energy_kev, pixel_m, distance_m,
					method='single', slope=None, alpha=0.0, envelope=None,
					reg_phi=(1e-6, 1e-10), reg_B=1e-10, lf_sigma=None,
					pad=True, air_ratio=None, chunk=32, workers=-1,
					dtype=np.float32):
	"""
	projections : (ny,nx), (n_ang,ny,nx), or for CTF (n_dist,n_ang,ny,nx)
					[or (n_dist,ny,nx)], flat-corrected intensities.
	distance_m  : scalar ('single') or sequence matching axis 0 (CTF modes).
	slope,alpha : material line delta = alpha + slope*mu  [slope in metres].
	envelope    : total thickness T (broadcastable to projection shape);
					only needed when alpha != 0 and phi output is wanted, and
					for the alpha-gradient data correction in 'single'.
	reg_phi     : (low-f, high-f) Tikhonov for phi in CTF mode; blended with a
					Gaussian of width lf_sigma (default: f where chi_max=0.05).
					Defaults suit low-noise data. Low-frequency phi is the
					ill-conditioned part of CTF retrieval: raise reg_phi[0]
					(e.g. 1e-3..1e-1) for noisy data, or use 'ctf_homogeneous'.
					Distance design rule: phase at feature frequency f_obj is
					well-conditioned only if z_max >~ 0.5/(pi*lambda*f_obj^2).
	air_ratio   : if set (e.g. 0.05), per-projection offset of phi and B fixed
					to zero over that fraction of columns at each side (air).
	Returns dict: B, phi (or None), and 'delta_proj' = Int(delta dz),
					'mu_proj' = Int(mu dz) -- FBP these to get delta(r), mu(r).
	"""
	lam = wavelength_m(energy_kev)
	P = np.asarray(projections, dtype)
	single = np.isscalar(distance_m)
	if pixel_m <= 0 or np.any(np.atleast_1d(np.asarray(distance_m, float)) <= 0):
		raise ValueError("pixel_m and all distances must be positive")
	zs = np.atleast_1d(np.asarray(distance_m, float))

	if method == 'single':
		if not single: raise ValueError("'single' takes one distance")
		if slope is None: raise ValueError("'single' needs slope (=delta/mu "
											"for one material, else line fit)")
		unwrap = (P.ndim == 2)
		stack = P[None] if unwrap else P
		if stack.ndim != 3:
			raise ValueError("'single' expects (ny, nx) or (n_ang, ny, nx)")
		ny, nx = stack.shape[-2:]
		py, px_, y0, x0 = _pad_setup(ny, nx, pad)
		F2 = _f2_grid(py, px_, pixel_m)
		den = 1.0 + 4*np.pi**2 * zs[0] * slope * F2
		if alpha != 0.0:
			if envelope is None:
				raise ValueError("'single' with alpha != 0 requires envelope: "
									"the alpha*grad(T) term is unmodelled "
									"otherwise (one-step defect correction)")
			env_full = np.broadcast_to(np.asarray(envelope, dtype), stack.shape)
		out = np.empty_like(stack)
		for i0 in range(0, stack.shape[0], chunk):
			blk = _pad(stack[i0:i0+chunk], py, px_, y0, x0)
			if alpha != 0.0:
				Tp = _pad(env_full[i0:i0+chunk], py, px_, y0, x0)
				I0a = np.clip(sfft.irfft2(sfft.rfft2(blk, workers=workers)/den,
											s=(py, px_), workers=workers), 1e-6, None)
				gy, gx = np.gradient(Tp, pixel_m, axis=(-2, -1))
				corr = (np.gradient(I0a*gx, pixel_m, axis=-1)
						+ np.gradient(I0a*gy, pixel_m, axis=-2))
				blk = blk - zs[0]*alpha*corr
			Ff = sfft.rfft2(blk, workers=workers) / den
			out[i0:i0+chunk] = sfft.irfft2(Ff, s=(py, px_),
											workers=workers)[..., y0:y0+ny, x0:x0+nx]
		B = -np.log(np.clip(out, 1e-6, None))
		if alpha == 0.0:
			phi = -(2*np.pi/lam) * slope * B
		else:
			phi = -(2*np.pi/lam) * (alpha*env_full + slope*B)

	elif method in ('ctf', 'ctf_homogeneous'):
		if alpha != 0.0 or envelope is not None:
			raise NotImplementedError(
				"affine inputs (alpha/envelope) are not supported in the "
				"LINEAR CTF modes; use alpha=0 here, or gauss_newton_refine"
				"(), which supports the affine model exactly.")
		if single or len(zs) < (2 if method == 'ctf' else 1):
			raise ValueError("CTF modes need a distance sequence "
								"(>=2 for 'ctf')")
		unwrap = (P.ndim == 3)
		stack = P[:, None] if unwrap else P               # (n_dist,n_ang,ny,nx)
		if stack.shape[0] != len(zs):
			raise ValueError("axis 0 of projections must match distances")
		n_ang, ny, nx = stack.shape[1:]
		py, px_, y0, x0 = _pad_setup(ny, nx, pad)
		F2 = _f2_grid(py, px_, pixel_m)
		chi = [np.pi*lam*z*F2 for z in zs]
		S, C = [np.sin(c) for c in chi], [np.cos(c) for c in chi]
		if method == 'ctf_homogeneous' or (method == 'ctf' and slope is not None):
			if slope is None: raise ValueError("'ctf_homogeneous' needs slope")
			coef = [-(2*(2*np.pi/lam)*slope*s_ + c_) for s_, c_ in zip(S, C)]
			denom = sum(c_*c_ for c_ in coef) + reg_B
		if method == 'ctf':
			Wb = np.exp(-F2/(2*(0.3/(np.pi*lam*zs.max()))))  # blend: chi_max=0.3
			if slope is not None:   # propagators for defect correction
				fyf = sfft.fftfreq(py, pixel_m)[:, None]
				fxf = sfft.fftfreq(px_, pixel_m)[None, :]
				Hc = [np.exp(-1j*np.pi*lam*z*(fyf**2+fxf**2)) for z in zs]
			if lf_sigma is None:
				lf_sigma = np.sqrt(0.05/(np.pi*lam*zs.max()))
			lphi = reg_phi[1] + (reg_phi[0]-reg_phi[1])*np.exp(-F2/(2*lf_sigma**2))
			a11 = sum(4*s_*s_ for s_ in S) + lphi
			a12 = sum(-2*s_*c_ for s_, c_ in zip(S, C))
			a22 = sum(c_*c_ for c_ in C) + reg_B
			det = a11*a22 - a12*a12
		phi = np.empty((n_ang, ny, nx), dtype)
		B = np.empty((n_ang, ny, nx), dtype)
		for i0 in range(0, n_ang, chunk):
			G = [sfft.rfft2(_pad(stack[d, i0:i0+chunk]-1.0, py, px_, y0, x0),
							workers=workers) for d in range(len(zs))]
			if method == 'ctf':
				if slope is not None:
					# nonlinear defect correction: subtract the
					# (nonlinear - linear) model mismatch at the prior
					FBh = sum(c_*g for c_, g in zip(coef, G))/denom
					Fph_h = -(2*np.pi/lam)*slope*FBh
					Bh = sfft.irfft2(FBh, s=(py, px_), workers=workers)
					phh = sfft.irfft2(Fph_h, s=(py, px_), workers=workers)
					u = np.exp(-Bh/2 + 1j*phh)
					U = sfft.fft2(u, workers=workers)
					for d in range(len(zs)):
						Inl = np.abs(sfft.ifft2(U*Hc[d], workers=workers))**2
						Ilin = 1.0 + sfft.irfft2(2*S[d]*Fph_h - C[d]*FBh,
													s=(py, px_), workers=workers)
						G[d] = G[d] - sfft.rfft2((Inl - Ilin).astype(dtype),
													workers=workers)
				b1 = sum(2*s_*g for s_, g in zip(S, G))
				b2 = sum(-c_*g for c_, g in zip(C, G))
				Fphi = (a22*b1 - a12*b2)/det
				FB = (a11*b2 - a12*b1)/det
				if slope is not None:
					Fphi = Wb*Fph_h + (1-Wb)*Fphi
					FB = Wb*FBh + (1-Wb)*FB
			else:
				FB = sum(c_*g for c_, g in zip(coef, G))/denom
				Fphi = -(2*np.pi/lam)*slope*FB
			phi[i0:i0+chunk] = sfft.irfft2(Fphi, s=(py, px_),
											workers=workers)[..., y0:y0+ny, x0:x0+nx]
			B[i0:i0+chunk] = sfft.irfft2(FB, s=(py, px_),
											workers=workers)[..., y0:y0+ny, x0:x0+nx]
	else:
		raise ValueError(f"unknown method {method!r}")

	if air_ratio:
		m = max(1, int(air_ratio*B.shape[-1]))
		for A in (B,) if phi is None else (B, phi):
			off = np.median(np.concatenate([A[..., :m], A[..., -m:]], -1),
							axis=(-2, -1), keepdims=True)
			A -= off
	if unwrap:                     # drop only the axis this function added
		B = B[0]
		phi = None if phi is None else phi[0]
	lam_ = wavelength_m(energy_kev)
	return dict(B=B, phi=phi, mu_proj=B,
				delta_proj=None if phi is None else -(lam_/(2*np.pi))*phi,
				method=method)

# ----------------------------------------------------------------------
def forward_project(image, angles_deg, pixel_m):
	"""Line integrals of a square 2D slice on the fbp() geometry.
	Contributions whose detector coordinate falls outside the grid are
	DISCARDED, not clamped into edge bins. Returns (n_ang, n)."""
	image = np.asarray(image, float)
	if image.ndim != 2 or image.shape[0] != image.shape[1]:
		raise ValueError("forward_project expects a square 2D slice")
	n = image.shape[0]
	xs = (np.arange(n) - n/2 + 0.5) * pixel_m
	Xg, Yg = np.meshgrid(xs, xs)
	flat = image.ravel()
	sino = np.zeros((len(angles_deg), n))
	for a, th in enumerate(np.deg2rad(angles_deg)):
		q = ((Xg*np.cos(th) + Yg*np.sin(th)).ravel() - xs[0]) / pixel_m
		i0 = np.floor(q).astype(int); w = q - i0
		acc = np.zeros(n)
		m0 = (i0 >= 0) & (i0 < n)
		m1 = (i0 >= -1) & (i0 < n - 1)
		np.add.at(acc, i0[m0], (flat*(1-w))[m0])
		np.add.at(acc, i0[m1] + 1, (flat*w)[m1])
		sino[a] = acc * pixel_m
	return sino

def classify_materials(mu_map, mu_table):
	"""Nearest-material voxel labels: 0 = air, j = mu_table[j-1]."""
	cents = np.concatenate([[0.0], np.asarray(mu_table, float)])
	return np.argmin(np.abs(mu_map[..., None] - cents), axis=-1)

def segment_materials(mu_map, mu_table, sigma=1.5, core_frac=0.3,
						core_erode=2):
	"""Artifact-robust material labels for a 2D slice: high-confidence CORES
	(values within core_frac of the class gaps, eroded) provide identity;
	watershed on the gradient magnitude propagates identities to the true
	edges. If a class has no core after erosion, erosion is relaxed
	adaptively; if it still has none, a ValueError names the class rather
	than silently deleting the material. Returns labels: 0 = air,
	j = mu_table[j-1]."""
	mt = np.asarray(mu_table, float)
	if (not np.all(np.isfinite(mt))) or np.any(mt <= 0):
		raise ValueError("mu_table must be finite and positive")
	if len(np.unique(mt)) != len(mt):
		raise ValueError("mu_table entries must be distinct")
	mu_lp = ndimage.gaussian_filter(mu_map, sigma)
	cents = np.concatenate([[0.0], mt])
	gaps = np.diff(np.sort(cents)); band = core_frac*gaps.min()
	markers = np.zeros(mu_map.shape, np.int32)
	for j, c in enumerate(cents):
		core0 = np.abs(mu_lp - c) < band
		core = core0
		for er in range(core_erode, -1, -1):
			core = (ndimage.binary_erosion(core0, iterations=er)
					if er else core0)
			if core.any():
				break
		if not core.any():
			raise ValueError(f"segment_materials: no core pixels for class "
								f"{j} (centroid {c:.4g}); check mu_table, "
								f"sigma, or region sizes")
		markers[core] = j + 1
	gm = ndimage.gaussian_gradient_magnitude(mu_lp, sigma)
	gm8 = (255*gm/max(gm.max(), 1e-30)).astype(np.uint8)
	lab = ndimage.watershed_ift(gm8, markers) - 1
	return np.clip(lab, 0, len(mt))

def predict_filter_error(psi, I0, energy_kev, pixel_m, distance_m, slope,
							pad=True, workers=-1):
	"""Validated backward-error of the slope-s single-distance filter:
	dB = (lam z / 2pi) * (1/I0) * L_s^{-1}[ div(I0 grad psi) ].
	psi, I0: (..., ny, nx). Subtract the result from the retrieved B."""
	lam = wavelength_m(energy_kev)
	psi = np.asarray(psi, float)
	if psi.ndim == 2: psi = psi[None]
	I0 = np.broadcast_to(I0, psi.shape)
	ny, nx = psi.shape[-2:]
	py, px_, y0, x0 = (_pad_setup(ny, nx, pad) if ny > 1
						else (1, _pad_setup(2, nx, pad)[1], 0,
								_pad_setup(2, nx, pad)[3]))
	Pp = _pad(psi, py, px_, y0, x0); Ip = _pad(I0, py, px_, y0, x0)
	gx = np.gradient(Pp, pixel_m, axis=-1)
	D = np.gradient(Ip*gx, pixel_m, axis=-1)
	if py >= 2:
		gy = np.gradient(Pp, pixel_m, axis=-2)
		D = D + np.gradient(Ip*gy, pixel_m, axis=-2)
	den = 1.0 + 4*np.pi**2*distance_m*slope*_f2_grid(py, px_, pixel_m)
	out = sfft.irfft2(sfft.rfft2(D, workers=workers)/den, s=(py, px_),
						workers=workers)[..., y0:y0+ny, x0:x0+nx]
	return (lam*distance_m/(2*np.pi)) * out / I0

def single_distance_multimaterial(projections, energy_kev, pixel_m, distance_m,
									angles_deg, mu_table, delta_table,
									weights=None, n_iter=1, rows=None,
									pad=True, air_ratio=0.05, omega=0.7,
									label_smooth=3, sino_sigma=1.0,
									classify_sigma=1.5):
	"""Single-distance retrieval for NON-collinear multi-material samples.
	Prior: each voxel is one of the tabulated materials (piecewise-constant
	3D), instead of collinearity. Pipeline per iteration: FBP -> classify ->
	forward-project class maps -> synthesize psi from off-line residuals ->
	subtract predict_filter_error from the ORIGINAL filtered retrieval.
	Iteration contract (correcting an earlier doc error): the update is
	C <- (1-omega)C + omega*dB, so n_iter=1 with the default omega=0.7
	applies a DELIBERATELY DAMPED PARTIAL correction (0.7*dB), not the
	frozen-label fixed point; omega=1.0 with n_iter=1 applies the full
	fixed-label correction in one step. The damping empirically hedges the
	tendency of the segmentation-synthesized correction to overshoot near
	interfaces. Re-segmenting corrected maps (n_iter>1) has no contraction
	guarantee and can drift when materials have opposite-sign eps; monitor
	the returned correction_norms and label_change history. Segmentation is core-identity
	watershed (segment_materials): interiors carry identity, gradient edges
	carry geometry -- robust to the interface-localized baseline artifact.
	Geometry: parallel-beam rows (use fresnel_scaling for cone data).
	Returns dict: B (corrected sinogram stack), mu (slices), labels,
	delta (classified delta slices), history_mu, line_fit."""
	fit = fit_material_line(mu_table, delta_table, weights, through_origin=True)
	s, eps = fit['slope'], fit['eps']
	lam = wavelength_m(energy_kev)
	r0 = retrieve_phase(projections, energy_kev, pixel_m, distance_m,
						method='single', slope=s, pad=pad, air_ratio=air_ratio)
	B0 = r0['B'] if r0['B'].ndim == 3 else r0['B'][None]
	rows = list(range(B0.shape[1])) if rows is None else list(rows)
	C = np.zeros((B0.shape[0], len(rows), B0.shape[2]))
	hist, corr_norms, label_change, prev_lab = [], [], [], None
	for it in range(n_iter + 1):
		Bc = B0.copy(); Bc[:, rows, :] = B0[:, rows, :] - C
		mu_sl = np.stack([fbp(Bc[:, r, :], angles_deg, pixel_m) for r in rows])
		lab = np.stack([segment_materials(m, mu_table, sigma=classify_sigma)
						for m in mu_sl])
		if label_smooth:
			lab = ndimage.median_filter(lab, size=(1, label_smooth, label_smooth))
		hist.append(mu_sl)
		if it == n_iter: break
		psi = np.zeros_like(C); Bsyn = np.zeros_like(C)
		for k in range(len(rows)):
			for j, (m_j, e_j) in enumerate(zip(mu_table, eps)):
				tj = forward_project((lab[k] == j+1).astype(float),
										angles_deg, pixel_m)
				if sino_sigma:
					tj = ndimage.gaussian_filter1d(tj, sino_sigma, axis=-1)
				psi[:, k, :] += -(2*np.pi/lam)*e_j*tj
				Bsyn[:, k, :] += m_j*tj
		dB = predict_filter_error(psi, np.exp(-Bsyn), energy_kev, pixel_m,
									distance_m, s, pad=pad)
		C = (1-omega)*C + omega*dB          # damped Picard update
		corr_norms.append(float(np.sqrt((C**2).mean())))
		if prev_lab is not None:
			label_change.append(float(np.mean(lab != prev_lab)))
		prev_lab = lab.copy()
	delta_sl = np.concatenate([[0.0], np.asarray(delta_table, float)])[lab]
	return dict(B=Bc, mu=mu_sl, labels=lab, delta=delta_sl,
				history_mu=hist, line_fit=fit,
				correction_norms=corr_norms, label_change=label_change)

# ----------------------------------------------------------------------
def extract_material_points(mu_map, delta_map, n_clusters, bins=256,
							smooth=2.0, trim_pct=80.0, n_iter=5):
	"""Harvest the material table from reconstructed (mu, delta) volumes:
	peaks of the 2D histogram -> trimmed nearest-centroid refinement
	(trimming suppresses partial-volume bridges between clusters).
	n_clusters should INCLUDE the air cluster. Returns dict with centroids
	(mu, delta) sorted by mu, covariances, and voxel fractions.
	Covariances are WITHIN-CLUSTER VOXEL SCATTER, not centroid standard
	errors: voxels are correlated by the reconstruction point spread, so
	do not divide by sqrt(N)."""
	x = np.ravel(np.asarray(mu_map, float))
	y = np.ravel(np.asarray(delta_map, float))
	xr = np.percentile(x, [0.1, 99.9]); yr = np.percentile(y, [0.1, 99.9])
	Hh, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
	Hs = ndimage.gaussian_filter(Hh, smooth)
	ismax = (Hs == ndimage.maximum_filter(Hs, size=9)) & (Hs > 0)
	idx = np.argwhere(ismax)
	order = np.argsort(Hs[ismax])[::-1]
	peaks = []
	for o in order:
		i, j = idx[o]
		c = (0.5*(xe[i]+xe[i+1]), 0.5*(ye[j]+ye[j+1]))
		if all(np.hypot((c[0]-p[0])/(xr[1]-xr[0]),
						(c[1]-p[1])/(yr[1]-yr[0])) > 0.12 for p in peaks):
			peaks.append(c)
		if len(peaks) == n_clusters: break
	if len(peaks) < n_clusters:
		raise ValueError(f"extract_material_points: found only {len(peaks)} "
							f"separable histogram peaks (requested "
							f"{n_clusters}); check n_clusters, smooth, or the "
							f"input maps")
	cents = np.array(peaks)
	sx, sy = max(np.std(x), 1e-30), max(np.std(y), 1e-30)
	for _ in range(n_iter):
		d2 = ((x[:, None]-cents[:, 0])/sx)**2 + ((y[:, None]-cents[:, 1])/sy)**2
		lab = np.argmin(d2, axis=1)
		for k in range(len(cents)):
			m = lab == k
			if not m.any(): continue
			r = d2[m, k]
			keep = r <= np.percentile(r, trim_pct)
			cents[k] = (x[m][keep].mean(), y[m][keep].mean())
	out_c, out_cov, out_fr = [], [], []
	d2 = ((x[:, None]-cents[:, 0])/sx)**2 + ((y[:, None]-cents[:, 1])/sy)**2
	lab = np.argmin(d2, axis=1)
	for k in np.argsort(cents[:, 0]):
		m = lab == k
		r = d2[m, k]; keep = r <= np.percentile(r, trim_pct)
		if int(keep.sum()) < 16:
			raise ValueError(f"extract_material_points: cluster at mu="
								f"{cents[k][0]:.4g} retains only "
								f"{int(keep.sum())} voxels; not usable")
		pts = np.stack([x[m][keep], y[m][keep]])
		out_c.append(cents[k]); out_cov.append(np.cov(pts))
		out_fr.append(m.mean())
	return dict(centroids=np.array(out_c), cov=out_cov,
				fractions=np.array(out_fr))

def infer_effective_energy(mu_measured, formulas, densities,
							e_range=(8.0, 80.0), n_grid=721):
	"""Fit the single effective energy that best explains measured effective
	attenuation coefficients of reference materials (e.g. holder inserts) on
	a polychromatic system. One free log-scale absorbs thickness/gain error.
	Returns (E_eff [keV], rms log-residual). The residual measures the
	irreducible inconsistency of a one-energy description of the spectrum
	and feeds the eps uncertainty budget. Requires xraylib."""
	import xraylib
	mu_measured = np.asarray(mu_measured, float)
	if not (len(mu_measured) == len(formulas) == len(densities)):
		raise ValueError("mu_measured, formulas, densities must match")
	if len(mu_measured) < 2:
		raise ValueError("need >=2 reference materials to infer energy")
	if np.any(mu_measured <= 0):
		raise ValueError("attenuation coefficients must be positive")
	L = np.log(mu_measured)
	Es = np.linspace(*e_range, n_grid)
	best = (np.inf, None)
	for E in Es:
		Lm = np.log([xraylib.CS_Total_CP(f, E)*r*100
						for f, r in zip(formulas, densities)])
		res = L - Lm; res -= res.mean()
		r2 = float(np.sum(res**2))
		if r2 < best[0]: best = (r2, E)
	return best[1], np.sqrt(best[0]/len(L))

def _fresnel_ops(B, phi, Hcs):
	"""Object wave and propagated waves for the scalar Fresnel model."""
	u0 = np.exp(-0.5*B + 1j*phi)
	U = sfft.fft2(u0)
	return u0, [sfft.ifft2(U*H) for H in Hcs]

def _fresnel_jvp(u0, uzs, Hcs, dB, dphi):
	"""Jacobian-vector product of I_z = |P_z exp(-B/2+i phi)|^2 at (B,phi)."""
	D = sfft.fft2(u0*(-0.5*dB + 1j*dphi))
	return [2.0*np.real(np.conj(uz)*sfft.ifft2(D*H))
			for uz, H in zip(uzs, Hcs)]

def _fresnel_vjp(u0, uzs, Hcs, rs):
	"""Adjoint of _fresnel_jvp for real residuals rs (one per distance)."""
	w = np.zeros_like(u0)
	for uz, H, r in zip(uzs, Hcs, rs):
		w += sfft.ifft2(sfft.fft2(2.0*uz*r)*np.conj(H))
	a = np.conj(u0)*w
	return -0.5*np.real(a), np.imag(a)

def gauss_newton_refine(projections, energy_kev, pixel_m, distances_m,
						B_init, phi_init, n_newton=2, n_cg=12, slope=None,
						alpha=0.0, envelope=None, prior_weight=1.0,
						reg_phi=1e-8, reg_B=1e-8, pad=True,
						dtype=np.float64):
	"""TRUE Gauss-Newton refinement of (B, phi) under the exact scalar
	Fresnel intensity model: each iteration linearizes the FULL nonlinear
	forward map at the CURRENT iterate (Jacobian-vector products, no
	weak-object assumption) and solves the regularized normal equations by
	preconditioned conjugate gradients (the weak-object CTF normal matrix
	is the exact Hessian at zero field and serves as a Fourier-diagonal
	preconditioner). This is the operation the linear path's "defect
	correction" approximates; unlike it, validity does not degrade with
	field strength, and the AFFINE material line rejected by the linear
	CTF modes is supported exactly: with alpha != 0 (envelope T required)
	the known component -k*alpha*T enters the initialization and the
	low-frequency prior penalty lam*Wb(f)*|phi + k(alpha*T + s*B)|^2.
	With slope=None and alpha=0 only small Tikhonov terms act -- the
	low-frequency phase null space of propagation data is a property of
	the DATA, and no solver removes it; the prior here is the same anchor
	as the linear blend, made explicit in the objective.

	residuals[a] holds the RMS nonlinear data residual per Newton
	iteration for projection a, PLUS the final value (n_newton+1 entries):
	the before/after reporting of the review's CORE-05. Cost is
	O(n_newton*n_cg) Fresnel propagator pairs per projection -- roughly an
	order beyond the linear solves. Fully coherent scalar model.
	Field shapes follow the retrieve_phase contract: (ny,nx) fields with
	(n_dist,ny,nx) projections, or (n_ang,ny,nx) with 4D projections.
	Iteration budget: the defaults suit weak-to-moderate fields; for
	strong fields (|phi| >~ 1 rad) the zero-field preconditioner weakens
	and more iterations are needed (e.g. n_newton=6, n_cg=40 converged a
	2.6 rad affine phantom); convergence is observable in residuals."""
	lam = wavelength_m(energy_kev)
	kp = 2*np.pi/lam
	P = np.asarray(projections, dtype)
	zs = np.atleast_1d(np.asarray(distances_m, float))
	if np.any(zs <= 0) or pixel_m <= 0:
		raise ValueError("distances and pixel_m must be positive")
	unwrap = (P.ndim == 3)
	stack = P[:, None] if unwrap else P
	if stack.ndim != 4 or stack.shape[0] != len(zs):
		raise ValueError("projections must be (n_dist[,n_ang],ny,nx) "
							"matching distances_m")
	B0 = np.asarray(B_init, dtype); p0 = np.asarray(phi_init, dtype)
	if unwrap:
		B0, p0 = B0[None], p0[None]
	n_ang, ny, nx = stack.shape[1:]
	if B0.shape != (n_ang, ny, nx) or p0.shape != (n_ang, ny, nx):
		raise ValueError("B_init/phi_init must match projection geometry")
	if alpha != 0.0 and envelope is None:
		raise ValueError("alpha != 0 requires envelope")
	env = (np.broadcast_to(np.asarray(envelope, dtype), (n_ang, ny, nx))
			if envelope is not None else None)
	py, px_, y0, x0 = _pad_setup(ny, nx, pad)
	fyf = sfft.fftfreq(py, pixel_m)[:, None]
	fxf = sfft.fftfreq(px_, pixel_m)[None, :]
	Hcs = [np.exp(-1j*np.pi*lam*z*(fyf**2 + fxf**2)) for z in zs]
	F2r = _f2_grid(py, px_, pixel_m)
	chi_r = [np.pi*lam*z*F2r for z in zs]
	use_prior = (slope is not None) or (alpha != 0.0)
	s_ = 0.0 if slope is None else slope
	if use_prior:
		Wb = np.exp(-F2r/(2*(0.3/(np.pi*lam*zs.max()))))
		def reg_prior(cB, cp, tpad):
			c = cp + kp*(s_*cB + (alpha*tpad if tpad is not None else 0.0))
			q = prior_weight*sfft.irfft2(Wb*sfft.rfft2(c), s=(py, px_))
			return kp*s_*q, q
	# Fourier-diagonal preconditioner: the weak-object CTF normal matrix
	# IS the GN Hessian at zero field, so it is near-exact for weak-to-
	# moderate objects and turns CG's crawl into a few iterations.
	m11 = sum(4*np.sin(c)**2 for c in chi_r) + reg_phi
	m12 = sum(-2*np.sin(c)*np.cos(c) for c in chi_r)
	m22 = sum(np.cos(c)**2 for c in chi_r) + reg_B
	if use_prior:
		m11 = m11 + prior_weight*Wb
		m12 = m12 + prior_weight*Wb*(kp*s_)
		m22 = m22 + prior_weight*Wb*(kp*s_)**2
	mdet = m11*m22 - m12**2 + 1e-30

	def precond(rB_, rp_):
		Fp = sfft.rfft2(rp_); FB = sfft.rfft2(rB_)
		zp = sfft.irfft2((m22*Fp - m12*FB)/mdet, s=(py, px_))
		zB = sfft.irfft2((-m12*Fp + m11*FB)/mdet, s=(py, px_))
		return zB, zp

	Bout = np.empty_like(B0); pout = np.empty_like(p0)
	hist = np.empty((n_ang, n_newton + 1))
	for a in range(n_ang):
		Ip = [_pad(stack[d, a], py, px_, y0, x0) for d in range(len(zs))]
		B = _pad(B0[a], py, px_, y0, x0)
		ph = _pad(p0[a], py, px_, y0, x0)
		tpad = _pad(env[a], py, px_, y0, x0) if env is not None else None
		for it in range(n_newton):
			u0, uzs = _fresnel_ops(B, ph, Hcs)
			rs = [I - np.abs(uz)**2 for I, uz in zip(Ip, uzs)]
			hist[a, it] = np.sqrt(np.mean([np.mean(r*r) for r in rs]))
			bB, bp = _fresnel_vjp(u0, uzs, Hcs, rs)
			bB -= reg_B*B; bp -= reg_phi*ph
			if use_prior:
				gB, gp = reg_prior(B, ph, tpad)
				bB -= gB; bp -= gp
			def Aop(dB, dp):
				JJb, JJp = _fresnel_vjp(u0, uzs, Hcs,
										_fresnel_jvp(u0, uzs, Hcs, dB, dp))
				JJb += reg_B*dB; JJp += reg_phi*dp
				if use_prior:
					gB2, gp2 = reg_prior(dB, dp, None)
					JJb += gB2; JJp += gp2
				return JJb, JJp
			xB = np.zeros_like(B); xp = np.zeros_like(ph)
			rB, rp = bB.copy(), bp.copy()
			zB, zp = precond(rB, rp)
			dB_, dp_ = zB.copy(), zp.copy()
			rz = float(np.sum(rB*zB) + np.sum(rp*zp))
			for _ in range(n_cg):
				qB, qp = Aop(dB_, dp_)
				den = float(np.sum(dB_*qB) + np.sum(dp_*qp))
				if den <= 0 or rz <= 0:
					break
				al = rz/den
				xB += al*dB_; xp += al*dp_
				rB -= al*qB; rp -= al*qp
				zB, zp = precond(rB, rp)
				rzn = float(np.sum(rB*zB) + np.sum(rp*zp))
				if rzn < 1e-24:
					break
				dB_ = zB + (rzn/rz)*dB_; dp_ = zp + (rzn/rz)*dp_
				rz = rzn
			B += xB; ph += xp
		u0, uzs = _fresnel_ops(B, ph, Hcs)
		hist[a, n_newton] = np.sqrt(np.mean(
			[np.mean((I - np.abs(uz)**2)**2) for I, uz in zip(Ip, uzs)]))
		Bout[a] = B[y0:y0+ny, x0:x0+nx]
		pout[a] = ph[y0:y0+ny, x0:x0+nx]
	if unwrap:
		Bout, pout = Bout[0], pout[0]
		hist = hist[0]
	return dict(B=Bout, phi=pout, residuals=hist)

def estimate_line_slope(projections, energy_kev, pixel_m, distances_m,
						s_range=(2e-10, 2e-8), n_scan=81, chi_band=(0.2, 4.0),
						angle_step=8, refine=2, pad=True, min_bins=32,
						dtype=np.float32):
	"""Prior-free material-line slope from a multi-distance stack, no
	reconstruction: fits the scalar s in the homogeneous CTF relation
	g_m = -(2(2pi/lam)s sin(chi_m) + cos(chi_m)) B on well-conditioned
	mid-frequency bins (chi_max within chi_band), then removes the
	quadratic (beyond-weak-object) intensity term by `refine` nonlinear
	defect-correction rounds around the homogeneous prior and re-fits.

	Raises ValueError if the frequency band is empty/too small for the
	given distances, or if the residual minimum sits on a scan boundary
	(unbracketed estimate) -- silent plausible numbers are worse than
	errors. Returns (s, info) with info containing s_grid, residual,
	explained_fraction (fraction of banded contrast captured by the
	single-slope model after correction: a DATA-DRIVEN collinearity
	statistic), n_bins, and n_angle_samples."""
	lam = wavelength_m(energy_kev)
	P = np.asarray(projections, dtype)
	if P.ndim != 4:
		raise ValueError("projections must be (n_dist, n_ang, ny, nx)")
	zs = np.atleast_1d(np.asarray(distances_m, float))
	if len(zs) < 2 or np.any(zs <= 0) or pixel_m <= 0:
		raise ValueError("need >=2 positive distances and positive pixel_m")
	if P.shape[0] != len(zs):
		raise ValueError("axis 0 of projections must match distances_m")
	ny, nx = P.shape[-2:]
	py, px_, y0, x0 = _pad_setup(ny, nx, pad)
	F2 = _f2_grid(py, px_, pixel_m)
	chi = [np.pi*lam*z*F2 for z in zs]
	band = (chi[np.argmax(zs)] > chi_band[0]) & (chi[np.argmax(zs)] < chi_band[1])
	nb = int(band.sum())
	if nb < min_bins:
		raise ValueError(
			f"estimate_line_slope: only {nb} frequency bins fall in "
			f"chi_band={chi_band} (chi_max reaches "
			f"{float(chi[np.argmax(zs)].max()):.3g}); increase the largest "
			f"distance, the band, or the grid")
	S = [np.sin(c) for c in chi]; C = [np.cos(c) for c in chi]
	fyf = sfft.fftfreq(py, pixel_m)[:, None]
	fxf = sfft.fftfreq(px_, pixel_m)[None, :]
	Hc = [np.exp(-1j*np.pi*lam*z*(fyf**2+fxf**2)) for z in zs]
	sel = slice(0, P.shape[1], angle_step)
	n_sel = len(range(*sel.indices(P.shape[1])))
	G = [sfft.rfft2(_pad(P[d, sel]-1.0, py, px_, y0, x0))
			for d in range(len(zs))]
	kp = 2*np.pi/lam
	ss = np.geomspace(*s_range, n_scan)

	def scan(Gb, final):
		Sb = [s_[band] for s_ in S]; Cb = [c_[band] for c_ in C]
		tot = sum(float(np.sum(np.abs(g)**2)) for g in Gb)
		res = np.empty(n_scan)
		for i, s in enumerate(ss):
			c = [-(2*kp*s*s_ + c_) for s_, c_ in zip(Sb, Cb)]
			num = np.abs(sum(cc*np.conj(g) for cc, g in zip(c, Gb)))**2
			den = sum(cc*cc for cc in c)
			res[i] = tot - float(np.sum(num/den))
		i = int(np.argmin(res))
		if i in (0, n_scan-1):
			# A boundary minimum in an intermediate scan is expected (the
			# nonlinear bias the refinement removes); only the RETURNED
			# estimate must be bracketed.
			if final:
				raise ValueError(
					f"estimate_line_slope: residual minimum at scan "
					f"boundary (s~{ss[i]:.3e}); unbracketed -- widen "
					f"s_range")
			return float(ss[i]), res, tot
		d1, d2 = res[i+1]-res[i-1], res[i+1]-2*res[i]+res[i-1]
		s_hat = float(ss[i])
		if d2 > 0:
			h = np.log(ss[1]/ss[0])
			s_hat = float(ss[i]*np.exp(-0.5*h*d1/d2))
		return s_hat, res, tot

	s_hat, res, tot = scan([g[..., band] for g in G], final=(refine == 0))
	for i_ref in range(refine):
		coef = [-(2*kp*s_hat*s_ + c_) for s_, c_ in zip(S, C)]
		den = sum(c_*c_ for c_ in coef) + 1e-12
		FB = sum(c_*g for c_, g in zip(coef, G))/den
		Fph = -kp*s_hat*FB
		Bh = sfft.irfft2(FB, s=(py, px_))
		phh = sfft.irfft2(Fph, s=(py, px_))
		U = sfft.fft2(np.exp(-Bh/2 + 1j*phh))
		Gc = []
		for d in range(len(zs)):
			Inl = np.abs(sfft.ifft2(U*Hc[d]))**2
			Ilin = 1.0 + sfft.irfft2(2*S[d]*Fph - C[d]*FB, s=(py, px_))
			Gc.append(G[d] - sfft.rfft2((Inl-Ilin).astype(dtype)))
		s_hat, res, tot = scan([g[..., band] for g in Gc],
								final=(i_ref == refine-1))
	info = dict(s_grid=ss, residual=res, n_bins=nb, n_angle_samples=n_sel,
				explained_fraction=(1.0 - float(res.min())/tot)
									if tot > 0 else 0.0)
	return s_hat, info

def harvest_material_table(projections, energy_kev, pixel_m, distances_m,
							angles_deg, n_materials, rows=None, slope=None,
							bootstrap=True, air_correct=True, pad=True,
							reg_phi=(1e-6, 1e-10), reg_B=1e-10,
							air_ratio=0.05, slope_sensitivity=None,
							n_folds=0):
	"""One call from a multi-distance scan to a material table, with the
	epistemic status of each output made explicit:

		mu_j    : DATA-DETERMINED (absorption is observable at all
				frequencies; centroids inherit the reconstruction's known
				discretization bias, recorded in provenance).
		delta_j : LOW-FREQUENCY-ANCHORED TO THE SLOPE PRIOR. Single-energy
				propagation data leave near-DC phase practically
				unobservable, so the blended CTF retrieval supplies it
				from the line delta = slope_used*mu. Material centroids
				are low-frequency quantities, hence harvested delta_j
				satisfy that relation largely BY CONSTRUCTION.
				Consequently line_fit['rms_eps'] MUST NOT be read as
				evidence that the sample is collinear. The data-driven
				collinearity statistic is 'collinearity_residual': the
				fraction of mid-frequency interface contrast NOT explained
				by the single best slope (prior-free, from
				estimate_line_slope).

	slope_sensitivity : optional float (e.g. 0.1). Repeats the harvest at
		slope_used*(1 -/+ value) and returns 'delta_slope_sensitivity'
		(approx. d delta / d ln s per material): the PRIOR SYSTEMATIC.

	n_folds : int, default 0. If >= 2, the harvest is repeated on n_folds
		interleaved angle subsets (angles[i::n_folds]), each re-deriving
		its own air reference and -- when the slope was estimated -- its
		own slope fit, so both propagate into the replicate scatter.
		Returned under 'uncertainty': fold points, fold slopes, and the
		per-material 2x2 centroid sample covariance across folds
		('centroid_cov_folds'). This is the STATISTICAL uncertainty
		(angular sampling, extraction, air reference). It does NOT contain
		the common-mode reconstruction bias (all folds share the FBP
		discretization bias) nor the slope-prior systematic (see
		slope_sensitivity); quote all three. Each replicate uses
		1/n_folds of the angles, so under var ~ 1/n_angles the full-data
		covariance is ~ centroid_cov_folds/n_folds_used
		('centroid_cov_full_est'); angular undersampling inflates
		replicate scatter, making this estimate conservative.

	Covariances in 'cov' remain within-cluster voxel scatter (PSF-
	correlated; not centroid standard errors). Requires an air/vacuum
	region in the field of view. Noisy data: supply a rough slope; real
	(non-periodic) data: keep pad=True with genuine air margins."""
	P = np.asarray(projections)
	if P.ndim != 4:
		raise ValueError("projections must be (n_dist, n_ang, ny, nx)")
	if len(angles_deg) != P.shape[1]:
		raise ValueError("angles_deg must match projections axis 1")
	angles_arr = np.asarray(angles_deg, float)
	ny = P.shape[2]
	if rows is None:
		rows = np.unique(np.linspace(0.3*ny, 0.7*ny, 5).astype(int))
	rows = list(rows)

	def _points(sl, P_, angs_):
		r = retrieve_phase(P_, energy_kev, pixel_m, distances_m,
							method='ctf', slope=sl, pad=pad,
							reg_phi=reg_phi, reg_B=reg_B, air_ratio=air_ratio)
		mu_sl = np.stack([fbp(r['mu_proj'][:, k, :], angs_, pixel_m)
							for k in rows])
		dl_sl = np.stack([fbp(r['delta_proj'][:, k, :], angs_, pixel_m)
							for k in rows])
		pts = extract_material_points(mu_sl, dl_sl, n_materials + 1)
		c = pts['centroids']
		i_air = int(np.argmin(np.abs(c[:, 0])))
		keep = np.ones(len(c), bool); keep[i_air] = False
		if abs(c[i_air, 0]) > 0.3*np.min(np.abs(c[keep, 0])):
			raise ValueError(
				"harvest_material_table: no clear air cluster (|mu_air| is "
				"not << material mu); ensure air is in the FOV or adjust "
				"n_materials")
		air_pt = c[i_air].copy()
		mu_m, dl_m = c[keep, 0].copy(), c[keep, 1].copy()
		if air_correct:
			mu_m -= air_pt[0]; dl_m -= air_pt[1]
		order = np.argsort(mu_m)
		cov = [pts['cov'][k] for k in np.where(keep)[0][order]]
		return (mu_m[order], dl_m[order], cov,
				pts['fractions'][keep][order], air_pt)

	slope_used, sinfo = slope, None
	if slope is None:
		if not bootstrap:
			raise ValueError("supply slope or enable bootstrap")
		slope_used, sinfo = estimate_line_slope(P, energy_kev, pixel_m,
												distances_m, pad=pad)
	mu_m, dl_m, cov, fr, air_pt = _points(slope_used, P, angles_arr)
	sens = None
	if slope_sensitivity:
		lo = _points(slope_used*(1 - slope_sensitivity), P, angles_arr)[1]
		hi = _points(slope_used*(1 + slope_sensitivity), P, angles_arr)[1]
		sens = (hi - lo)/(2*slope_sensitivity)
	unc = None
	if n_folds >= 2:
		fm, fd, fs = [], [], []
		for i in range(n_folds):
			Pi, ai = P[:, i::n_folds], angles_arr[i::n_folds]
			sl_i = slope_used
			if slope is None:
				try:
					sl_i, _ = estimate_line_slope(
						Pi, energy_kev, pixel_m, distances_m, pad=pad,
						angle_step=max(1, int(round(8/n_folds))))
				except ValueError:
					pass                      # keep full-data slope
			try:
				mi, di = _points(sl_i, Pi, ai)[:2]
			except ValueError:
				continue                      # fold unusable; count below
			if len(mi) == n_materials:
				fm.append(mi); fd.append(di); fs.append(sl_i)
		if len(fm) < 2:
			raise ValueError("harvest_material_table: fewer than 2 angle "
								"folds produced a usable table; reduce "
								"n_folds or supply more angles")
		fm = np.array(fm); fd = np.array(fd)
		covs = [np.cov(np.stack([fm[:, j], fd[:, j]]), ddof=1)
				for j in range(n_materials)]
		unc = dict(n_folds=n_folds, n_folds_used=len(fm),
					fold_mu=fm, fold_delta=fd, slope_folds=np.array(fs),
					centroid_cov_folds=covs,
					centroid_cov_full_est=[c_/len(fm) for c_ in covs])
	fit = fit_material_line(mu_m, dl_m, through_origin=True)
	return dict(mu=mu_m, delta=dl_m, cov=cov, fractions=fr,
				air_point=air_pt, line_fit=fit, slope_used=slope_used,
				delta_prior_informed=True,
				collinearity_residual=(None if sinfo is None else
										1.0 - sinfo['explained_fraction']),
				delta_slope_sensitivity=sens, uncertainty=unc,
				provenance=dict(energy_kev=energy_kev,
								distances_m=list(np.atleast_1d(distances_m)),
								rows=rows, air_correct=air_correct,
								slope_source=('estimated' if slope is None
												else 'supplied'),
								slope_info=sinfo))

def transfer_table(mu, delta, e_from, e_to,
					basis=(("H2O", 1.00), ("Al", 2.70))):
	"""Transfer harvested material points to another energy, valid only
	when no constituent (or basis material) has an absorption edge between
	e_from and e_to, AND the materials are representable in the chosen
	two-basis expansion (necessary-but-not-sufficient: check the returned
	weights). delta: exact E^-2 scaling. mu: Alvarez-Macovski two-basis
	expansion pinned by the harvested (mu, delta) pair. Requires xraylib.
	Returns (mu_new, delta_new, info) where info reports basis weights,
	row-scaled condition number, and an extreme-weight flag."""
	import xraylib
	mu = np.asarray(mu, float); delta = np.asarray(delta, float)
	def point(f, r, E):
		return (xraylib.CS_Total_CP(f, E)*r*100.0,
				1.0 - xraylib.Refractive_Index_Re(f, E, r))
	A_from = np.array([point(f, r, e_from) for f, r in basis]).T
	A_to = np.array([point(f, r, e_to) for f, r in basis]).T
	rn = np.linalg.norm(A_from, axis=1, keepdims=True)
	cond = float(np.linalg.cond(A_from/rn))
	w = np.linalg.solve(A_from, np.stack([mu, delta]))
	info = dict(weights=w, basis_condition=cond,
				extreme_weights=bool(np.any(np.abs(w) > 10.0)))
	return A_to[0] @ w, delta*(e_from/e_to)**2, info

# ----------------------------------------------------------------------
def fbp(sinogram, angles_deg, pixel_m):
	"""Minimal parallel-beam FBP (Ram-Lak). sinogram: (n_ang, nx).
	Returns (nx, nx) map in the integrand's units (e.g. mu_proj -> mu [1/m])."""
	sinogram = np.asarray(sinogram, float)
	if sinogram.ndim != 2 or sinogram.shape[0] != len(angles_deg):
		raise ValueError("sinogram must be (n_ang, nx) with n_ang matching "
							"angles_deg")
	n_ang, nx = sinogram.shape
	npad = sfft.next_fast_len(2*nx)
	f = np.abs(sfft.fftfreq(npad, pixel_m))
	q = sfft.ifft(sfft.fft(np.pad(sinogram, ((0, 0), (0, npad-nx))), axis=1)*f,
					axis=1).real[:, :nx]
	xs = (np.arange(nx) - nx/2 + 0.5) * pixel_m
	Xg, Yg = np.meshgrid(xs, xs)
	rec = np.zeros((nx, nx))
	for a, th in enumerate(np.deg2rad(angles_deg)):
		u = Xg*np.cos(th) + Yg*np.sin(th)
		rec += np.interp(u, xs, q[a], left=0, right=0)
	return rec * np.pi / n_ang
