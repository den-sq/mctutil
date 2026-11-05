# Propagation-based phase retrieval

This is the supplied **as-built reference implementation**, imported into
`mctutil.phase`. Its numerical checks use exact scalar Fresnel propagation of
noiseless synthetic objects. **Material-table harvesting is experimental.**
Real-data accuracy, noise, detector blur, partial coherence and polychromatic
illumination remain unvalidated. The source document was dated 2026-08-31.

## 1. Target and scope

Retrieve absorption projections `B` and phase `phi` before tomography, harvest
material tables from co-registered multi-distance scans, and apply known tables
to single-distance multimaterial correction. The existing functions own the
scientific calculations; packaging and command-line adapters do not reimplement
them. This import does not provide a validated material library.

## 2. Physical model and conventions

The refractive index is `n = 1 - delta + i*beta`, with
`mu = 4*pi*beta/lambda`, `B = integral(mu dz)`, and
`phi = -(2*pi/lambda)*integral(delta dz)`. Positive `delta` therefore gives
negative phase. Energies are in keV; core API lengths and pixels are in metres;
`mu` is in inverse metres and `delta` is dimensionless. The composition helper
accepts stock density in g/cm³.

Inputs are flat/dark-corrected intensities with unit background, registered on
one effective pixel grid. `(ny, nx)` and `(angles, ny, nx)` retain their shape,
including `ny=1`; multi-distance input `(distances, angles, ny, nx)` returns an
angle stack. `fresnel_scaling(R1, R2, pixel_det)` returns effective distance,
magnification and effective pixel size. Multi-distance cone-beam input must
first be magnification-rescaled to a common grid.

## 3. Reference implementation contract

The import preserves scientific expressions, defaults, generated test data and
assertion tolerances. Indentation, including docstrings, follows the repository's tabs convention;
reference formatting and complexity have scoped lint exceptions to avoid an
algorithmic rewrite during import. The material helper's import is adapted to
the package. Reference scripts have explicit entry points and local plot output
paths, so importing them does not execute a validation campaign.

## 4. The material line

`fit_material_line` fits `delta = alpha + slope*mu` and returns the off-line
residuals `eps`. A through-origin collinear material set admits the
single-distance filter within the transport-of-intensity approximation. A
nonzero intercept needs a known thickness envelope and uses an approximate
one-step correction. `materials.build_table` computes points from composition,
measured stock density and energy using xraylib's total attenuation cross
sections, and returns the line fit and per-material artifact budget.

## 5. Linear retrieval (`retrieve_phase`)

- `single`: generalized Paganin filtering; nonzero `alpha` requires `envelope`.
- `ctf_homogeneous`: CTF inversion constrained to a material-line slope.
- `ctf`: coupled phase/absorption inversion from at least two distances. With a
  slope, low frequencies use a material-line prior and one nonlinear defect
  subtraction. This correction is not Gauss–Newton.

The linear CTF paths reject affine inputs. Use consistent boundary conditions:
the supplied periodic simulations use `pad=False` where specified; real
non-periodic inputs require suitable air margins and padding. Regularization
and `dtype` remain caller choices. Return keys are `B`, `phi`, `delta_proj` and
`mu_proj`.

## 6. Exact-model refinement (`gauss_newton_refine`)

The refiner uses exact Fresnel Jacobian-vector products and conjugate gradients.
It returns `B`, `phi` and per-projection `residuals` before and after each Newton
step. The affine prior requires a known envelope. Strong phase objects can
need larger `n_newton` and `n_cg` budgets; inspect the residuals. Refinement does
not recover the unobservable phase DC component or remove prior dependence.

## 7. Material-table workflow

`estimate_line_slope` fits a slope from multi-distance mid-frequency contrast
and returns `(slope, info)`, including `explained_fraction`. Harvesting combines
that estimate with CTF retrieval, row-wise FBP, cluster extraction,
air-referencing and a material-line fit. `harvest_material_table` returns
`mu`, `delta`, covariance, fractions, air point, fit, uncertainty and provenance;
`delta_prior_informed=True` explicitly records the prior's role.

`single_distance_multimaterial` uses a known table, reconstruction and
segmentation to predict and subtract filter error. Its default is **one damped
step with omega=0.7**. Further segmentation/correction iterations have no
contraction guarantee; inspect `correction_norms` and `label_change`.
`transfer_table` and `infer_effective_energy` use xraylib. Transfer assumes
matching composition and density without absorption edges in the energy band.

## 8. Observability

Absorption centroids are data-determined. Harvested low-frequency `delta` is
largely anchored to the supplied or estimated material-line prior, so harvested
`line_fit['rms_eps']` is **not evidence of material collinearity**. The
mid-frequency `collinearity_residual` is the data-driven statistic. A single
energy does not independently determine each material's low-frequency
off-line deviation. Independently characterized materials or additional energy
information are needed to resolve that ambiguity.

## 9. Uncertainty

Keep angle-fold statistical covariance, slope-prior sensitivity and common-mode
reconstruction bias separate. Fold scatter cannot expose a bias shared by all
folds. Cluster covariance is within-cluster voxel scatter, not a centroid
standard error. The reference harvest inherits FBP discretization bias; this
import does not correct it or reinterpret it as statistical noise.

## 10. Validation and reference figures

The four supplied validation programs generate their numerical inputs; no
external scan data are required. Only the compact assertion suite is collected
by pytest. The larger programs print reference measurements rather than impose
new acceptance tolerances. From the repository root, with the package installed
(or `PYTHONPATH=.`):

```sh
python -m pytest tests/phase/test_phase_tomo_unit.py
python tests/phase/validate_phase_tomo.py /tmp/phase-validation
python tests/phase/validate_multimat_single.py /tmp/phase-validation
python tests/phase/validate_gn.py
python examples/phase/harvest_table_demo.py
```

The numerical module requires NumPy and SciPy. Plotted validations additionally
require Matplotlib. The composition helper and energy-transfer portion of the
demo require xraylib; the demo reports when optional transfer is skipped.

The original figures are preserved byte-for-byte:
[Tier 0](../../examples/phase/tier0_validation.png),
[multipart](../../examples/phase/multipart_validation.png),
[tomography](../../examples/phase/tomo_validation.png), and
[multimaterial](../../examples/phase/multimat_single.png).
The archive includes no generating scripts for the Tier 0 or multipart figures;
they are supplied reference evidence, not independently reproduced tests.
Some supplied plot labels retain older terminology; the corrections in §13
take precedence, including the distinction between defect correction and GN.

## 11. Known limitations

The scalar coherent forward model does not model detector blur, photon noise,
registration error, partial coherence or a polychromatic spectrum. Effective
energy is a calibration device, not a spectral model. Material harvesting
remains **experimental and unvalidated on real data**. Interface accuracy is
limited by segmentation and pixelation, and FBP bias remains. Feeding
`gauss_newton_refine` outputs into `single_distance_multimaterial` is explicitly
**unvalidated**. Importing this package does not close those validation gates.

## 12. Package and provenance

Import numerical functions from `mctutil.phase.phase_tomo` and the composition
helper from `mctutil.phase.materials`. For example:

```python
from mctutil.phase.phase_tomo import retrieve_phase, gauss_newton_refine

initial = retrieve_phase(projections, energy_kev, pixel_m, distances_m,
                         method="ctf", slope=slope)
refined = gauss_newton_refine(projections, energy_kev, pixel_m, distances_m,
                              initial["B"], initial["phi"], slope=slope)
```

The four CLI commands take a JSON configuration and an output path:

```sh
mctutil phase harvest scan.json harvested.json
mctutil phase estimate-line-slope scan.json slope.json
mctutil phase refine refinement.json refined.npz
mctutil phase make-material-table compositions.json materials.json
```

All four commands support `--dry-run` to report the configuration and output
paths without loading arrays, running calculations or writing results.

Each configuration contains keyword arguments for the corresponding function:
`harvest_material_table`, `estimate_line_slope`, `gauss_newton_refine` or
`materials.build_table`. Include only arguments that function accepts. Array
arguments `projections`, `B_init`, `phi_init`, `envelope` and `angles_deg` accept
JSON arrays or `.npy` paths resolved relative to the configuration file. For
example, a harvest configuration can contain:

```json
{
  "projections": "projections.npy",
  "angles_deg": "angles.npy",
  "energy_kev": 20.0,
  "pixel_m": 5e-8,
  "distances_m": [0.004, 0.010, 0.025],
  "n_materials": 3,
  "slope_sensitivity": 0.1,
  "n_folds": 4
}
```

JSON results retain `schema_version`, `producer`, the original `parameters`,
`config_path` and `result`; the slope result contains `slope` and `info`.
Refinement writes an NPZ containing `B`, `phi`, `residuals` and scalar JSON
`metadata` with the same provenance envelope. These are computation records;
the experimental and prior-dependence qualifications above still apply.

Source: user-supplied `phase_tomo_package.zip`, SHA-256
`a990e033adcfeca4bbd51eeeddbc43c3a860c36ae6f65e38a084e970ff3b8b94`.
The numerical module has an identical Python AST to the archive source after
excluding docstring indentation. The table helper changes only its import;
validation/demo adaptations change only
imports, execution wrapping and plot destinations. This README distills the
source document; §13 below is reproduced verbatim.

| Archive file | Original SHA-256 |
| --- | --- |
| `phase_tomo.py` | `9986cf8c3fa9df9d250a902d2d197fe51f0849eb66aadc17eed3f9284ed71ab0` |
| `make_material_table.py` | `fc35cd97f7c73d393d041be7e588ce9c25df898b88e9b064454bdf5d199666cd` |
| `test_phase_tomo_unit.py` | `da11451e2199d247904824b52edc4d84129968648f16b396fdba9e19a0884b3b` |
| `test_phase_tomo.py` | `06808fbb98599de873c4b70da19f7c0cce1cfe92dbc6bc3918da6d3d8d4fa532` |
| `test_multimat_single.py` | `b6298c44679c06840a1702d81597dda1e9f64823016ef315cb3c58268df4afad` |
| `validate_gn.py` | `644f462fd133870c31b97393750367bebdbe242ae6e08c0abc77f0beb3534428` |
| `harvest_table_demo.py` | `3be3a8f435f3ccb10d5b31dc121620dff29a309060010da5319bbd7d0269e944` |


## 13. Documentation corrections log

Recorded so stale claims are not trusted downstream. The affine
single-distance path was previously described as exact; it is a one-step
defect correction, and the envelope is now mandatory when α ≠ 0. The
nonlinear correction in the linear paths was previously called
"Gauss–Newton"; it is Born-type residual subtraction with linear
contraction, and the true Gauss–Newton is §6. The multimaterial driver's
"one step is the fixed point" claim contradicted its ω = 0.7 default; the
iteration contract is now stated correctly, including the measured
overshoot the damping compensates. Harvested δ was previously presented as
validated to −0.8…−2.6% without qualification; that accuracy is real for a
collinear sample but inherited from the prior being correct, and the
epistemic split of §8 now accompanies every harvested table.
