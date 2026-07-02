"""
beam_tracking.py  (revised)
===========================
Time-resolved flatfield (gain) tracking for X-ray tomography when the beam
background drifts between the pre-scan and post-scan flats while static
scintillator defects do NOT move.

Physical model
--------------
	pre (x) = D(x) * B0(x)             # static defects D  x  beam at t=0
	post(x) = D(x) * B1(x)             # static defects D  x  beam at t=1
	R   (x) = post/pre = B1/B0         # ratio -> D cancels exactly (multiplicative)

Goal: the gain at projection time t in [0,1]:   G(x,t) = D(x) * B(x,t)

WHAT CHANGED IN THIS REVISION (and why)
---------------------------------------
Earlier guidance called this problem "noise-limited" and down-weighted motion
modelling. That was WRONG for properly-prepared flats, for two reasons:

	* Median-combined flats are clean. Measured white-noise floor on the example
	data is ~0.18% (at the 8-bit quantisation step), not the ~8% a single-shot
	Poisson assumption implies. diagnose() now MEASURES the noise floor
	(Immerkaer, per-patch minimum) instead of inferring it from counts.

	* The high-frequency part of log(post/pre) is strongly spatially CORRELATED
	(autocorr ~0.87 at lag 1), i.e. real beam texture that MOVED, not white
	noise. diagnose() now reports this correlated-motion fraction and routes
	accordingly: when noise << motion, the warp-based methods (3,4) are worth
	using; crossfade leaves ~0.4-0.7% ghosting on fine moving bands.

	* 8-bit round-trips (e.g. via ImageJ/PowerPoint) mainly hurt DARK regions,
	where few quant levels remain (rel. noise 1-3% at the frame bottom).
	diagnose() flags bit depth and dark-region degradation. Reprocess from the
	original 12-bit median'd flats for best results.

Methods (increasing physical sophistication)
---------------------------------------------
	1. geometric_crossfade        - log-linear interp. Baseline; ghosts moving
									sharp/fine structure (~0.5% here).
	2. optimal_transport_interp   - SAFE DEFAULT. Crossfade's static-noise
									averaging + a denoised transport ramp.
									Captures bulk motion + intensity drift with no
									flow-estimation / aliasing risk.
	3. flow_warp_interp           - RECOMMENDED on clean (12-bit) data. Small-
									displacement, single-scale, alias-guarded ratio
									Horn-Schunck flow -> FRACTIONAL warp. Recovers
									the fine-band ghosting method 2 leaves behind.
	4. separated_warp_interp      - full model: static D split off and held fixed;
									only the smooth beam is warped. Use when motion
									is large AND prominent defects are present
									(prevents defect smearing / dipole artefacts).
	5. keyframe_interp            - piecewise interpolation through a TIME-SERIES
									of flat snapshots (flat_series_digest.py
									output). REQUIRED when the beam leaves the
									pre->post line: beam events, energy recovery,
									non-monotonic drift. Methods 1-4 cannot
									represent those states with ANY schedule.

Nonlinear drift: every two-flat method accepts schedule, a callable s(t)
mapping scan-fraction to beam-time. This handles nonlinear PACE along the
pre->post line only; if the beam leaves that line (see a measured event series:
72% of frames >1 px off-line), use keyframe_interp instead.

All methods return a callable  G(t) -> ndarray. Use flatfield_projection(...).
"""
from __future__ import annotations
from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import LOG, log

try:
	from scipy.ndimage import gaussian_filter, map_coordinates, zoom
except ImportError:
	def _missing_scipy(*_args, **_kwargs):
		raise RuntimeError("scipy is required for flat beam tracking; install mctutil[flats].")

	gaussian_filter = map_coordinates = zoom = _missing_scipy


def _require_scipy():
	if getattr(gaussian_filter, "__name__", "") == "_missing_scipy":
		raise click.ClickException("scipy is required for flat beam tracking; install mctutil[flats].")


def _require_tifffile():
	try:
		import tifffile
	except ImportError as exc:
		raise click.ClickException(
			"tifffile is required for flat beam tracking; install mctutil[flats]."
		) from exc
	return tifffile


# --------------------------------------------------------------------------- #
#  utilities
# --------------------------------------------------------------------------- #
def _clean(a, eps=None):
	a = np.asarray(a, np.float64).copy()
	bad = ~np.isfinite(a) | (a <= 0)
	if bad.any():
		good = a[~bad]
		a[bad] = np.median(good) if eps is None else eps
	return a


def _grid(shape):
	H, W = shape
	yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
	return yy, xx


def _warp(img, v, u, yy, xx):
	return map_coordinates(img, [yy + v, xx + u], order=1, mode="nearest")


def _as_schedule(schedule):
	"""Return a vectorised s(t) in [0,1]; identity if None."""
	if schedule is None:
		return lambda t: t
	return schedule


# --------------------------------------------------------------------------- #
#  noise-floor estimator (measured, not assumed)
# --------------------------------------------------------------------------- #
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)


def estimate_noise_floor(img, patch=64):
	"""Robust white-noise std (ADU). Immerkaer response per patch; the smoothest
	patches (low percentile) give the noise floor, immune to real structure."""
	try:
		from scipy.signal import convolve2d
	except ImportError as exc:
		raise RuntimeError("scipy is required for flat beam tracking; install mctutil[flats].") from exc
	a = _clean(img)
	H, W = a.shape
	vals = []
	for i in range(0, H - patch, patch):
		for j in range(0, W - patch, patch):
			blk = a[i:i + patch, j:j + patch]
			if blk.mean() <= 1:
				continue
			c = convolve2d(blk, _IMMERKAER, mode="valid")
			vals.append(np.sqrt(np.pi / 2) / (6 * c.size) * np.sum(np.abs(c)))
	if not vals:
		return np.nan
	return float(np.percentile(vals, 10))


def _bit_depth_info(img):
	a = _clean(img)
	u = np.unique(a)
	is_int = np.allclose(a, np.round(a))
	step = float(np.min(np.diff(u))) if u.size > 1 else np.nan
	levels = int(u.size)
	eight_bit = bool(is_int and a.max() <= 255 and levels <= 256)
	return dict(integer=bool(is_int), step=step, levels=levels,
				eight_bit=eight_bit, vmax=float(a.max()))


# --------------------------------------------------------------------------- #
#  diagnostics  (revised routing)
# --------------------------------------------------------------------------- #
def diagnose(pre, post, beam_sigma=8, dark_quantile=0.15):
	"""Characterise the beam change and recommend a method, from MEASURED noise
	and motion content (not assumed Poisson)."""
	pre, post = _clean(pre), _clean(post)
	bd = _bit_depth_info(pre)
	lp, lq = np.log(pre), np.log(post)
	lR = lq - lp
	lo = gaussian_filter(lR, beam_sigma)
	hf = lR - lo

	sig = estimate_noise_floor(pre)
	bright = pre[pre > np.percentile(pre, 60)].mean()
	noise_rel = (sig / bright) if (sig and bright) else np.nan
	logR_noise = noise_rel * np.sqrt(2)

	h = hf - hf.mean()
	denom = np.mean(h * h)
	lag1 = np.mean(h[:, :-1] * h[:, 1:]) / denom if denom > 0 else 0.0
	fine_motion = float(np.sqrt(max(np.var(hf) - logR_noise**2, 0)))

	dark_thr = np.quantile(pre, dark_quantile)
	dark_rel_q = (bd["step"] / max(dark_thr, 1e-6)) if bd["integer"] else 0.0

	info = dict(
		intensity_drift_pct=float((np.exp(np.median(lR)) - 1) * 100),
		noise_floor_ADU=float(sig), noise_rel_pct=float(noise_rel * 100),
		logR_smooth_std=float(np.std(lo)),
		logR_highfreq_std=float(np.std(hf)),
		highfreq_lag1_autocorr=float(lag1),
		fine_motion_std=fine_motion,
		motion_to_noise=float(np.std(lo) / max(logR_noise, 1e-9)),
		bit_depth=bd, dark_rel_quant_pct=float(dark_rel_q * 100),
	)

	log.write(
		"Bit Depth",
		(
			f"{'8-bit round-trip' if bd['eight_bit'] else 'OK'} "
			f"({bd['levels']} levels, step {bd['step']:.3g}, max {bd['vmax']:.0f})"
		),
	)
	log.write("Intensity Drift", f"{info['intensity_drift_pct']:+.1f}%")
	log.write("Noise Floor", f"{sig:.3f} ADU ({info['noise_rel_pct']:.2f}% of signal)")
	log.write("Smooth Motion", f"{info['logR_smooth_std']:.4f} logR")
	log.write(
		"High-Freq LogR",
		(
			f"{info['logR_highfreq_std']:.4f}; lag-1 autocorr {lag1:.2f} -> "
			f"{'moved structure' if lag1 > 0.3 else 'noise-like'}"
		),
	)
	log.write("Fine Motion", f"{fine_motion:.4f}; noise part {logR_noise:.4f}")
	log.write("Dark Quant", f"{info['dark_rel_quant_pct']:.1f}% frame-bottom relative noise")

	motion_dominated = info["motion_to_noise"] > 3 and (lag1 > 0.3 or fine_motion > 2*logR_noise)
	if not motion_dominated:
		log.write("Recommendation", "noise comparable to motion -> optimal_transport_interp (method 2)")
	elif bd["eight_bit"] or dark_rel_q > 0.01:
		log.write(
			"Recommendation",
			"motion-dominated, but 8-bit/dark degradation present; use method 2 or reprocess 12-bit flats for methods 3/4",
		)
	else:
		log.write(
			"Recommendation",
			"clean and motion-dominated -> flow_warp_interp (3) or separated_warp_interp (4)",
		)
	return info


def recommend(pre, post):
	"""Return the suggested method factory name as a string."""
	info = diagnose(pre, post)
	if info["motion_to_noise"] <= 3:
		return "optimal_transport_interp"
	if info["bit_depth"]["eight_bit"] or info["dark_rel_quant_pct"] > 1.0:
		return "optimal_transport_interp"
	return "flow_warp_interp"


# --------------------------------------------------------------------------- #
#  Method 1 - geometric (log-linear) crossfade  [baseline]
# --------------------------------------------------------------------------- #
def geometric_crossfade(pre, post, schedule=None):
	"""G(t) = pre^(1-s) * post^s,  s = schedule(t)."""
	pre, post = _clean(pre), _clean(post)
	lp, lq = np.log(pre), np.log(post)
	s = _as_schedule(schedule)
	return lambda t: np.exp((1.0 - s(t)) * lp + s(t) * lq)


# --------------------------------------------------------------------------- #
#  Method 2 - noise-optimal transport interpolation   [SAFE DEFAULT]
# --------------------------------------------------------------------------- #
def optimal_transport_interp(pre, post, transport_sigma=12, schedule=None):
	"""G(t) = exp( (logpre+logpost)/2 + (s-1/2) * LowPass(logpost-logpre) )."""
	pre, post = _clean(pre), _clean(post)
	lp, lq = np.log(pre), np.log(post)
	avg = 0.5 * (lp + lq)
	S = gaussian_filter(lq - lp, transport_sigma)
	s = _as_schedule(schedule)
	return lambda t: np.exp(avg + (s(t) - 0.5) * S)


# --------------------------------------------------------------------------- #
#  Method 3 - small-displacement, alias-guarded flow + fractional warp
# --------------------------------------------------------------------------- #
def estimate_flow_ratio_HS(pre, post, beam_sigma=4, env_sigma=40,
							smoothness=2.0, n_iter=300, prefilter=1.0,
							downsample=4, max_disp=None, verbose=True):
	"""Ratio-driven Horn-Schunck on the envelope-normalised, defect-suppressed
	beam texture. Solves  logR ~= -(u . grad logB)  (defects cancel in R).
	Estimates on a downsampled grid (fast, denoises), single-scale, zero-init,
	with optional max_disp clamp to flag band aliasing."""
	pre, post = _clean(pre), _clean(post)
	H, W = pre.shape
	d = max(int(downsample), 1)
	lp, lq = np.log(pre), np.log(post)
	if d > 1:
		lp = zoom(lp, 1 / d, order=1)
		lq = zoom(lq, 1 / d, order=1)
	bs, es = beam_sigma, env_sigma / d
	tp = gaussian_filter(lp, bs) - gaussian_filter(lp, es)
	tq = gaussian_filter(lq, bs) - gaussian_filter(lq, es)
	It = tq - tp
	Ix = 0.5 * (np.gradient(tp, axis=1) + np.gradient(tq, axis=1))
	Iy = 0.5 * (np.gradient(tp, axis=0) + np.gradient(tq, axis=0))
	u = np.zeros_like(It); v = np.zeros_like(It)
	a2 = smoothness ** 2
	for _ in range(n_iter):
		ub = gaussian_filter(u, prefilter); vb = gaussian_filter(v, prefilter)
		num = Ix * ub + Iy * vb + It
		den = a2 + Ix ** 2 + Iy ** 2
		u = ub - Ix * num / den
		v = vb - Iy * num / den
	if d > 1:
		v = zoom(v, (H / v.shape[0], W / v.shape[1]), order=1) * d
		u = zoom(u, (H / u.shape[0], W / u.shape[1]), order=1) * d
	md = float(np.percentile(np.hypot(v, u), 99))
	if verbose:
		log.write(
			"Flow",
			f"median v={np.median(v):+.2f} u={np.median(u):+.2f} px, 99th-pct |disp|={md:.2f} px",
		)
	if max_disp is not None and md > max_disp:
		log.write(
			"Flow Clamp",
			f"|disp| {md:.1f}px > max_disp {max_disp}px; possible band aliasing, clamping",
			log_level=LOG.WARN,
		)
		v *= max_disp / md; u *= max_disp / md
	return v, u


def flow_warp_interp(pre, post, beam_sigma=4, transport_sigma=12,
						schedule=None, **flow_kw):
	"""Fractional-warp interpolation using small-displacement flow. Static
	defects stay fixed; smooth beam warps by s*(v,u); residual intensity ramps."""
	pre, post = _clean(pre), _clean(post)
	yy, xx = _grid(pre.shape)
	lp, lq = np.log(pre), np.log(post)
	v, u = estimate_flow_ratio_HS(pre, post, beam_sigma=beam_sigma, **flow_kw)
	logB0 = gaussian_filter(lp, beam_sigma)
	Dstatic = lp - logB0
	B0w_full = _warp(logB0, v, u, yy, xx)
	drift = gaussian_filter(lq - B0w_full - Dstatic, transport_sigma)
	s = _as_schedule(schedule)

	def G(t):
		st = s(t)
		B0w = _warp(logB0, st * v, st * u, yy, xx)
		return np.exp(Dstatic + B0w + st * drift)
	G.flow = (v, u)
	return G


# --------------------------------------------------------------------------- #
#  Method 4 - full static/dynamic separation + fractional warp
# --------------------------------------------------------------------------- #
def separated_warp_interp(pre, post, beam_sigma=4, transport_sigma=12,
							use_flow=True, schedule=None, **flow_kw):
	"""Explicit split of static defects from the moving beam; D0 held fixed so
	defects never smear. Degrades to a denoised transport ramp if use_flow=False."""
	pre, post = _clean(pre), _clean(post)
	yy, xx = _grid(pre.shape)
	lp, lq = np.log(pre), np.log(post)
	logB0 = gaussian_filter(lp, beam_sigma)
	logD0 = lp - logB0
	S = gaussian_filter(lq - lp, transport_sigma)
	s = _as_schedule(schedule)
	if use_flow:
		v, u = estimate_flow_ratio_HS(pre, post, beam_sigma=beam_sigma, **flow_kw)
	else:
		v = u = 0.0

	def G(t):
		st = s(t)
		if use_flow:
			B0w = _warp(logB0, st * v, st * u, yy, xx)
			residual = st * (S - (B0w - logB0))
			return np.exp(logD0 + B0w + residual)
		return np.exp(logD0 + logB0 + st * S)
	if use_flow:
		G.flow = (v, u)
	return G


# --------------------------------------------------------------------------- #
#  Method 5 - keyframe interpolation from a flat TIME-SERIES
# --------------------------------------------------------------------------- #
#  WHY: every method above interpolates between exactly two flats, so it can only
#  produce beam states ON the pre->post line. Real drift can leave that line
#  entirely: on a measured 10-min flat series containing a beam event (intensity
#  dip + ~3 px vertical sag, then slow non-monotonic recovery), the beam state
#  was >1 px off the endpoint line for 72% of the series (median 2.3 px, max
#  4.6 px). NO schedule can fix that -- the states simply aren't representable.
#  With ~20 keyframes (60-frame spacing) the piecewise model's median error
#  collapsed to 0.04 px; the digest produced by flat_series_digest.py is exactly
#  such a keyframe stack.
def keyframe_interp(flats, times, transport_sigma=12):
	"""Piecewise gain model from a sequence of clean flat snapshots (keyframes).

	flats : (n, H, W) array or list of n flats, time-ordered (e.g. the
			digest_stack.tif from flat_series_digest.py -- small median
			snapshots, NOT broad averages).
	times : n scan-fractions in [0,1] (or any monotone axis, e.g. frame index
			or seconds; queries use the same axis, normalised or not).
	transport_sigma : smoothing of each pair's transport field (as in
			optimal_transport_interp).

	Returns G(t): finds the bracketing keyframe pair (k, k+1) and applies the
	noise-optimal pairwise interpolation
		log G = (logF_k + logF_{k+1})/2 + (w - 1/2) * LowPass(logF_{k+1}-logF_k)
	with w the position of t inside the interval. Static defects D appear in
	every keyframe, so they remain fixed automatically. Handles arbitrary,
	non-monotonic beam trajectories -- the beam just has to pass THROUGH the
	keyframes, not move steadily between two endpoints.

	Queries outside [times[0], times[-1]] clamp to the nearest interval
	(extrapolation via the end pair). Per-pair transport fields are computed
	lazily and one pair is cached, which is efficient when projections are
	processed in time order.
	"""
	times = np.asarray(times, float)
	if times.ndim != 1 or len(times) < 2:
		raise ValueError("need at least 2 keyframes with 1D `times`")
	if np.any(np.diff(times) <= 0):
		raise ValueError("`times` must be strictly increasing")
	n = len(times)
	logs = [np.log(_clean(f)) for f in flats]      # keep logs; flats can be freed
	if len(logs) != n:
		raise ValueError("len(flats) != len(times)")

	cache = {"k": None, "avg": None, "S": None}

	def _pair(k):
		if cache["k"] != k:
			la, lb = logs[k], logs[k + 1]
			cache["avg"] = 0.5 * (la + lb)
			cache["S"] = gaussian_filter(lb - la, transport_sigma)
			cache["k"] = k
		return cache["avg"], cache["S"]

	def G(t):
		k = int(np.searchsorted(times, t) - 1)
		k = min(max(k, 0), n - 2)                  # clamp -> end-pair extrapolation
		w = (t - times[k]) / (times[k + 1] - times[k])
		avg, S = _pair(k)
		return np.exp(avg + (w - 0.5) * S)

	G.times = times
	return G


# --------------------------------------------------------------------------- #
#  static / dynamic DECOMPOSITION -> separate defect and beam flats
# --------------------------------------------------------------------------- #
#  Model per flat k:   F_k(x) = D(x) * B_k(x)
#     D   : static multiplicative artefact field (scintillator defects, dust,
#           fibre-optic chicken wire, ...) -- values ~1, does NOT move.
#     B_k : dynamic beam background at time k -- envelope + moving fine bands.
#
#  IDENTIFIABILITY (important): with a single pair of flats, any fine structure
#  that did NOT move between them is ambiguous -- it could be detector or
#  stationary beam texture. The temporal escape: D is stable across ALL frames,
#  while moving beam texture decorrelates at a fixed pixel as it shifts. Hence
#     log D = temporal MEDIAN over k of highpass(log F_k)
#  converges to the true static field as soon as the series contains beam
#  motion larger than the band width. Feed this a flat time-series (the digest
#  stack) whenever possible; with only pre/post and sub-pixel motion, unmoved
#  beam bands will be classified as "static" (harmless for correction --
#  dividing them out per-projection is identical either way -- but not a clean
#  semantic separation). Broad static nonuniformity (scale > beam_sigma) is
#  likewise inseparable from the beam envelope and is assigned to B.
def decompose_static_dynamic(flats, beam_sigma=4, band_orientation="horizontal",
								band_length=75, exclude_mask=None):
	"""Split flats into a static defect field D and dynamic beam flats B_k.

	flats : one 2D flat, a (pre, post) pair, or an (n, H, W) stack / list
			(time-ordered digest snapshots are the best input).
	beam_sigma : scale (px) separating "fine" (candidate defect) structure from
			the smooth beam envelope. Defects narrower than the beam bands are
			captured; structure broader than this goes to B.
	band_orientation : 'horizontal', 'vertical', or None. Synchrotron beam
			texture is typically stripe-like (horizontal in both example
			datasets); a running median of length `band_length` ALONG the
			stripe direction isolates that elongated texture per frame and
			assigns it to the beam -- even when it has not moved. Compact
			defects are untouched by the long directional median. Set None to
			rely on temporal stability alone (then adequate beam MOTION,
			larger than the band wavelength, is required for a clean split;
			elongated static defects such as scratches along the stripe
			direction will be misassigned to the beam when this is enabled).
	band_length : window (px) of the directional median; should be several x
			the defect size and << the frame width.
	exclude_mask : optional boolean mask of pixels to leave out of the static
			estimate (e.g. a sample region); D=1 there.

	Estimator (log space, per frame k):
		hf_k   = log F_k - G_sigma(log F_k)          # fine structure
		band_k = directional_median(hf_k)            # elongated beam stripes
		logD   = temporal_median_k( hf_k - band_k )  # what is stable AND compact
	B_k = F_k / D exactly, so D * B_k reconstructs each input flat.

	Identifiability: compact fine structure that never moved AND any elongated
	static structure (with band_orientation set) are the residual ambiguities;
	more frames and more motion shrink the former.
	"""
	from scipy.ndimage import median_filter
	arr = np.asarray(flats, np.float64)
	if arr.ndim == 2:
		arr = arr[None]
	n = arr.shape[0]
	size = ((1, band_length) if band_orientation == "horizontal"
			else (band_length, 1) if band_orientation == "vertical" else None)
	# temporally-stable, compact high-frequency component = static defects
	hf_stack = np.empty(arr.shape, np.float32)
	for k in range(n):                       # loop keeps peak memory ~2 frames
		lf = np.log(_clean(arr[k]))
		hf = lf - gaussian_filter(lf, beam_sigma)
		if size is not None:                 # remove elongated beam stripes
			hf = hf - median_filter(hf, size=size)
		hf_stack[k] = hf.astype(np.float32)
	logD = np.median(hf_stack, axis=0) if n > 1 else hf_stack[0]
	del hf_stack
	if exclude_mask is not None:
		logD = logD.copy()
		logD[exclude_mask] = 0.0
	D = np.exp(logD).astype(np.float32)
	B = np.stack([( _clean(arr[k]) / D).astype(np.float32) for k in range(n)])
	return D, B


def save_decomposition(flats, out_dir, names=None, beam_sigma=4, **decomp_kw):
	"""Run decompose_static_dynamic and write the results as TIFFs:

		<out_dir>/static_defects.tif        (D  -- divide out once, applies to
													every frame of the scan)
		<out_dir>/dynamic_beam_<name>.tif   (B_k -- one per input flat; use
													these with any interpolator)

	names : optional labels per flat (default: their index). Returns the list
	of written file paths. Note D * B_k == F_k exactly, so correcting with
	(D, B_k) is algebraically identical to correcting with F_k -- the value of
	the split is INDEPENDENT handling: D can be divided out of projections
	once (or inspected / QC'd / reused across scans with the same detector),
	while beam tracking operates on the B_k alone.
	"""
	tifffile = _require_tifffile()
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	D, B = decompose_static_dynamic(flats, beam_sigma=beam_sigma, **decomp_kw)
	if names is None:
		names = [str(k) for k in range(B.shape[0])]
	paths = [out_dir / "static_defects.tif"]
	tifffile.imwrite(paths[0], D)
	for k, nm in enumerate(names):
		p = out_dir / f"dynamic_beam_{nm}.tif"
		tifffile.imwrite(p, B[k])
		paths.append(p)
	return paths


# --------------------------------------------------------------------------- #
#  beam-time calibration from projection AIR regions
# --------------------------------------------------------------------------- #
#  In sample-free (air) pixels a projection IS the beam background (x ~const
#  transmission), so the beam-time of each projection can be fitted directly:
#  find t minimising the band-scale mismatch of  log(proj) - [(1-t)logpre+t logpost].
#  This matters because the first projection is generally NOT at t=0 (the pregain
#  was captured earlier) -- on the example data beam_t ran 0.63 -> 1.18, not 0->1.
def _bandpass(x, lo=1.5, hi=25.0):
	return gaussian_filter(x, lo) - gaussian_filter(x, hi)


def auto_air_mask(proj, pre, ratio_hi=0.95, exposure_quantile=0.5):
	"""Boolean mask of sample-free, well-exposed pixels: proj/pre near 1 (no
	attenuation) and pre bright enough to avoid the dark frame edges."""
	proj, pre = _clean(proj), _clean(pre)
	ratio = proj / pre
	expo = pre > np.quantile(pre, exposure_quantile)
	return (ratio > ratio_hi) & expo


def fit_beam_time(proj, pre, post, mask=None, band=(1.5, 25.0)):
	"""Closed-form beam-time t* of a single projection from its air region:
		t* = <a, s> / <s, s>,  a = BP(log proj - log pre), s = BP(log post - log pre)
	where BP is a band-pass isolating beam structure from white noise & smooth
	tilt. Returns t* (0 ~ pregain state, 1 ~ postgain state; may lie outside)."""
	pre, post, proj = _clean(pre), _clean(post), _clean(proj)
	if mask is None:
		mask = auto_air_mask(proj, pre)
	lpre, lpost = np.log(pre), np.log(post)
	s = _bandpass(lpost - lpre, *band)
	a = _bandpass(np.log(proj) - lpre, *band)
	m = mask & np.isfinite(a) & np.isfinite(s)
	sm = s[m]
	denom = float(np.sum(sm * sm))
	if denom <= 0:
		return float("nan")
	return float(np.sum(a[m] * sm) / denom)


def calibrate_schedule(projs, scan_fractions, pre, post, mask=None,
						kind="linear", band=(1.5, 25.0), verbose=True):
	"""Fit beam-time vs scan-fraction from several timestamped projections and
	return (schedule, info). `schedule(t)` maps scan-fraction t in [0,1] to the
	calibrated beam-time to pass through the interpolators, e.g.:

		s, info = calibrate_schedule(projs, fracs, pre, post)
		G = optimal_transport_interp(pre, post, schedule=s)
		corrected = flatfield_projection(proj_k, G, t=k/(n-1))

	`projs`: list of projection arrays (or a shared air `mask` may be supplied).
	`scan_fractions`: each projection's index/(n-1).
	`kind`: 'linear' (robust; extrapolates) or 'interp' (piecewise, monotone)."""
	fr = np.asarray(scan_fractions, float)
	if mask is None:
		mask = auto_air_mask(projs[0], pre)
	tstar = np.array([fit_beam_time(p, pre, post, mask=mask, band=band)
						for p in projs])
	order = np.argsort(fr); fr, tstar = fr[order], tstar[order]
	if kind == "linear":
		A = np.vstack([np.ones_like(fr), fr]).T
		alpha, beta = np.linalg.lstsq(A, tstar, rcond=None)[0]
		sched = lambda t: alpha + beta * np.asarray(t, float)
		resid = float(np.std(tstar - (alpha + beta * fr)))
		info = dict(alpha=float(alpha), beta=float(beta), resid=resid,
					tstar=tstar.tolist(), scan_fractions=fr.tolist())
	else:
		ti = np.maximum.accumulate(tstar)        # enforce monotone
		sched = lambda t: np.interp(t, fr, ti)
		info = dict(tstar=tstar.tolist(), scan_fractions=fr.tolist())
	if verbose:
		log.write("Calibration", "beam-time calibration from air regions")
		for f, ti in zip(fr, tstar):
			log.write("Beam Time", f"scan-frac {f:.3f} -> beam_t {ti:+.3f}")
		if kind == "linear":
			log.write(
				"Linear Fit",
				f"beam_t = {info['alpha']:+.3f} + {info['beta']:.3f}*scan_frac (resid {info['resid']:.3f})",
			)
			log.write(
				"Schedule",
				f"first projection already at beam_t {info['alpha']:.2f}; pass schedule= into interpolators",
				log_level=LOG.WARN,
			)
	return sched, info


# --------------------------------------------------------------------------- #
#  fitting a nonlinear motion schedule from a flat time-series
# --------------------------------------------------------------------------- #
def schedule_from_trajectory(frac, motion, kind="cumulative"):
	"""Build s(t) from measured drift. frac=scan-fraction in [0,1];
	motion=cumulative drift magnitude (e.g. centroid path length). Returns a
	monotone s(t) with s(0)=0, s(1)=1.

	CAUTION: kind='cumulative' enforces monotonicity (np.maximum.accumulate),
	which SILENTLY DISTORTS non-monotonic trajectories -- e.g. a beam event
	where the beam sags and recovers. If the measured drift reverses direction,
	do not use this at all: a schedule only re-paces motion ALONG the pre->post
	line, and an event takes the beam OFF that line. Use keyframe_interp with
	the digest snapshots instead."""
	frac = np.asarray(frac, float); motion = np.asarray(motion, float)
	order = np.argsort(frac); frac, motion = frac[order], motion[order]
	m = motion - motion[0]
	m = np.maximum.accumulate(m) if kind == "cumulative" else m
	if m[-1] == 0:
		return lambda t: t
	m = m / m[-1]
	return lambda t: np.interp(t, frac, m)


# --------------------------------------------------------------------------- #
#  real-data validation: is beam-drift tracking even worth it on this scan?
# --------------------------------------------------------------------------- #
def beam_only_residual_decomp(pre, post, proj, t, mask, smooth=8):
	"""Decide whether pre/post drift-tracking can help a given scan, using an
	intermediate projection that has a SAMPLE-FREE (beam-only) region.

	In beam-only pixels, proj ~= true gain at fraction t. We measure the residual
	structure left by each gain model and, crucially, what fraction of the
	residual lies ALONG the pre->post drift direction -- i.e. the most any
	two-flat interpolation could ever remove. The remainder is smooth
	envelope/scatter mismatch between scan-time and flat-time (a different
	problem: dynamic/sample-aware flatfield).

	pre, post, proj : 2-D arrays (same shape)
	t        : scan fraction of `proj` in [0,1]
	mask     : bool array, True on trustworthy beam-only pixels
	Returns dict of residual stds and `frac_along_drift` in [0,1].
	"""
	pre, post, proj = _clean(pre), _clean(post), _clean(proj)
	m = np.asarray(mask, bool)
	lpost = np.log(post)
	drift = gaussian_filter(lpost - np.log(pre), smooth)
	r = gaussian_filter(np.log(proj) - lpost, smooth)
	d = drift[m] - drift[m].mean()
	rr = r[m] - r[m].mean()
	frac_along = float(np.dot(rr, d) ** 2 / (np.dot(d, d) * np.dot(rr, rr) + 1e-12))

	def sres(G):
		q = gaussian_filter(np.log(proj) - np.log(_clean(G)), smooth)
		return float(np.std(q[m] - q[m].mean()))

	out = dict(
		static_pre=sres(pre), static_post=sres(post),
		crossfade=sres(geometric_crossfade(pre, post)(t)),
		optimal=sres(optimal_transport_interp(pre, post)(t)),
		frac_along_drift=frac_along,
	)
	best = min(("static_pre", "static_post", "crossfade", "optimal"),
				key=lambda k: out[k])
	out["best"] = best
	out["tracking_worthwhile"] = bool(
		frac_along > 0.3 and
		out["crossfade"] < 0.9 * min(out["static_pre"], out["static_post"]))
	return out


# --------------------------------------------------------------------------- #
#  applying the tracking map to projections
# --------------------------------------------------------------------------- #
def flatfield_projection(proj, G, t, dark=0.0):
	"""corrected = (proj - dark) / (G(t) - dark);  t = index/(n-1)."""
	g = G(t)
	return (np.asarray(proj, np.float64) - dark) / np.clip(g - dark, 1e-6, None)


def build_gain_stack(G, n, dark=0.0):
	return np.stack([G(i / (n - 1)) for i in range(n)], axis=0)


@click.command()
@click.argument("pre_flat", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("post_flat", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"--digest-stack",
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	help="Time-ordered flat digest stack to use for decomposition.",
)
@click.option(
	"--decompose-output",
	type=click.Path(file_okay=False, path_type=Path),
	help="Write static_defects.tif and dynamic_beam_*.tif to this directory.",
)
@click.option("--beam-sigma", type=float, default=4.0, show_default=True, help="Gaussian scale separating beam from defects.")
@click.option(
	"--band-orientation",
	type=click.Choice(["horizontal", "vertical", "none"]),
	default="horizontal",
	show_default=True,
	help="Stripe orientation to suppress while estimating static defects.",
)
@click.option("--band-length", type=click.IntRange(1), default=75, show_default=True, help="Directional median window length.")
@click.option("--dry-run", is_flag=True, help="Plan decomposition writes without writing files.")
def beam_tracking(pre_flat, post_flat, digest_stack, decompose_output, beam_sigma, band_orientation, band_length, dry_run):
	"""Diagnose beam drift and optionally split flat fields into static/dynamic components."""
	_require_scipy()
	log.start()

	if (pre_flat is None) != (post_flat is None):
		raise click.UsageError("Provide both PRE_FLAT and POST_FLAT, or neither.")
	if pre_flat is None and digest_stack is None:
		raise click.UsageError("Provide PRE_FLAT POST_FLAT and/or --digest-stack.")

	tifffile = _require_tifffile()
	flats_for_decomposition = None
	names = None

	if pre_flat is not None and post_flat is not None:
		pre = tifffile.imread(pre_flat)
		post = tifffile.imread(post_flat)
		log.write("Input", f"pre={pre_flat} post={post_flat}")
		diagnose(pre, post)
		flats_for_decomposition = np.stack([pre, post])
		names = ["pre", "post"]

	if digest_stack is not None:
		digest = np.asarray(tifffile.imread(digest_stack))
		if digest.ndim == 2:
			digest = digest[None]
		log.write("Digest Stack", f"{digest_stack}: {digest.shape[0]} frame(s) {digest.shape[1:]}")
		flats_for_decomposition = digest
		names = [f"{index:04d}" for index in range(digest.shape[0])]

	if decompose_output is None:
		return

	orientation = None if band_orientation == "none" else band_orientation
	if dry_run:
		log.write("Dry Run", f"Would write {Path(decompose_output) / 'static_defects.tif'}")
		for name in names or []:
			log.write("Dry Run", f"Would write {Path(decompose_output) / f'dynamic_beam_{name}.tif'}")
		return

	paths = save_decomposition(
		flats_for_decomposition,
		decompose_output,
		names=names,
		beam_sigma=beam_sigma,
		band_orientation=orientation,
		band_length=band_length,
	)
	for path in paths:
		log.write("File Written", str(path))


if __name__ == "__main__":
	beam_tracking()
