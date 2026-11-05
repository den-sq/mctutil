# Material-table export contract, version 1

`mctutil phase export CONFIG_JSON OUTPUT` reads a JSON producer record made by
`phase harvest` or `phase make-material-table`. It writes a JSON artifact for
another repository to consume without importing mctutil. These are boundary
adapters around the supplied numerical functions, not a new runtime model.
`--dry-run` reports the planned paths without reading the calibration or
computing a transfer.

An export configuration for parallel geometry is:

```json
{
  "calibration": "harvest.json",
  "effective_energy_kev": 24.0,
  "geometry": {"kind": "parallel", "pixel_m": 0.000001, "distance_m": 0.1},
  "envelope_m": null
}
```

The calibration path is relative to the export configuration. Cone geometry
instead accepts exactly `kind: "cone"`, `source_to_sample_m`,
`sample_to_detector_m`, and `detector_pixel_m`. It contains native physical
inputs only; magnification, effective distance and effective pixel size are
derived by the consumer's Fresnel geometry facility.

Optional `transfer_basis` is a two-element list of `[formula, density_g_cm3]`
pairs, defaulting to `[["H2O", 1.0], ["Al", 2.7]]`. It applies only when the
requested energy differs from calibration energy. The existing `transfer_table`
owns transfer calculations and diagnostics. The existing `fit_material_line`
then fits the target points with the producer's original weights and
through-origin policy. At unchanged energy, the producer's fit is retained.

`envelope_m` is null, a scalar, or nested JSON arrays describing total spatial
thickness T in metres, broadcastable to the eventual projection shape. This is
the reference function's envelope, not a Fourier weighting envelope. Nonzero
line intercept requires an envelope. Shape compatibility is checked when the
consumer has the projections; the export does not invent an envelope model or
load additional arrays.

## Artifact fields and units

| Field | Meaning and units |
| --- | --- |
| `schema` | Exactly `mctutil.phase.material_table`. |
| `schema_version` | Exactly `1`. Consumers must reject unsupported versions. |
| `effective_energy_kev` | Energy of the exported material points, in keV. |
| `geometry` | The exclusive native parallel or cone input object above; lengths in metres. |
| `envelope_m` | Spatial total thickness in metres, or null. |
| `material_table.names` | Ordered material names from the composition producer; null for unidentified harvested classes. |
| `material_table.mu` | One-dimensional ordered attenuation coefficients, m⁻¹, at the effective energy. |
| `material_table.delta` | Matching dimensionless refractive-index decrements at the effective energy. |
| `material_table.line_fit` | `alpha` (dimensionless intercept), `slope` (metres), `eps` (ordered dimensionless residuals), `rms_eps` (dimensionless). The line is δ = α + slope × μ. |
| `calibration.producer` | `harvest_material_table` or `build_table`. |
| `calibration.energy_kev` | Source calibration energy, keV. |
| `calibration.config_path` | Original producer configuration path, for provenance. |
| `calibration.parameters` | Remaining producer inputs after the boundary normalization below. These describe the source calibration, not target geometry. |
| `calibration.provenance` | Harvest provenance, including resolved distances in metres, rows, air referencing and slope source/info; null for composition tables. |
| `calibration.observations` | Available `fractions`, `air_point`, `slope_used` and `collinearity_residual` from the harvest. |
| `calibration.uncertainty` | The source-energy budget below; never silently transported to target energy. |
| `transfer` | Null at unchanged energy; otherwise basis plus the reference function's `weights`, `basis_condition`, `extreme_weights`. |
| `status` | Harvested tables remain `experimental`; composition tables are `model_derived`. |
| `provenance` | Explicit origins of calibration μ/δ, target points, line fit and ε, as described below. |
| `target_uncertainty` | `see_calibration_budget` at unchanged energy; `not_propagated` after transfer. |

There is exactly one operative material table and fit. The source main μ/δ
arrays, source fit, and derived plateau artifact budget are not copied into
`calibration`. Fold samples remain there because they are uncertainty evidence,
not an alternative operative table. Material indices preserve the producer's
ordering; harvesting does not identify chemical compositions.

The input record is not modified. At the export boundary, source energy is
normalized to `calibration.energy_kev` and omitted from copied parameters and
provenance; conflicting harvest input/provenance energies are rejected. Resolved
harvest distances, rows and air referencing live only in provenance. The
operative prior is `observations.slope_used`, with its supplied/estimated origin
in `provenance.slope_source`; the redundant input slope is omitted. Requested
fold count is retained in the statistical budget when that budget exists;
otherwise the input setting can remain in parameters. The input
`delta_prior_informed` flag is consumed into the single
`provenance.calibration_delta` label, rather than copied as another carrier.

For harvest observations, fractions are dimensionless; `air_point` has
coordinates `[mu, delta]` in `[m⁻¹, 1]`; `slope_used` is in metres and is the
retrieval prior, not the target fitted slope. `collinearity_residual` is the
dimensionless measured mid-frequency unexplained fraction; it can be null when
the slope was supplied.
Slope-fit diagnostics in provenance retain their reference units: slope grid
in metres, dimensionless residuals and explained fraction, and integer counts.

## Uncertainty and epistemic status

`calibration.uncertainty.scope` is always `calibration_energy`, referring to
`calibration.energy_kev`. Its fields preserve distinct quantities:

- `statistical`: the original fold budget or null. It includes `n_folds`,
  `n_folds_used`, `fold_mu` (m⁻¹), `fold_delta` (dimensionless), `slope_folds`
  (metres), and per-material `centroid_cov_folds` / `centroid_cov_full_est`.
  Covariance coordinates are `[mu, delta]`, with matrix units
  `[[m⁻², m⁻¹], [m⁻¹, 1]]`. The full-data estimate uses the reference's
  variance-versus-angle-count assumption; it is not a new estimator.
- `delta_slope_sensitivity`: the original dimensionless dδ/d(ln slope) values
  or null. This is prior sensitivity, not an already scaled error bar.
- `voxel_scatter_cov`: original per-cluster covariance matrices or null,
  with the same coordinate units. These are correlated within-cluster voxel
  scatter, **not centroid standard errors**.
- `common_mode_bias`: `not_quantified`. The source's fold procedure cannot
  measure common reconstruction bias. The measured air point remains in
  observations; reference-demo FBP bias percentages are not measurements of
  the exported calibration.

Null means unavailable or not requested, never zero uncertainty. Composition
tables have no quantified uncertainty from the supplied producer. Transfer
does not propagate any of these quantities: source fold samples, covariances
and sensitivity remain at source energy even if target coefficients change.

The `provenance` fields are explicit labels. `calibration_mu` is
`measured_absorption` for harvest, while `calibration_delta` is
`slope_prior_informed`. Both are
`xraylib_from_supplied_composition_density` for composition tables.
`calibration_mu_bias` is `fbp_discretization_bias_not_removed` for harvest
and `not_applicable` for composition tables; no numerical bias estimate is
invented.
`target_points` is `calibration_points` or `energy_transfer_model`;
`line_fit` is `derived_from_target_points`; and `eps` is
`prior_informed_not_independently_measured` for harvest or
`derived_from_model_points` for composition tables. A small harvested fit
residual is not evidence of collinearity: use the separate data-derived
mid-frequency statistic where available.

Harvest exports remain experimental pending mctutil #198's noise/reference
validation gates. The Gauss–Newton-to-multimaterial composition remains
unvalidated; this writer accepts neither refinement records nor claims of
validation. Energy transfer assumes no constituent or basis absorption edge
between energies and representability in the two chosen basis materials.
The returned condition and extreme-weight flag are diagnostics, not proof of
those assumptions. No new scientific certification is added by exporting.
Harvest inputs must retain their provenance, prior flag and uncertainty fields;
bare coefficient arrays cannot be turned into a labeled calibration by this
writer. Explicit null uncertainty estimates remain acceptable.

## Mapping to lftomo

The inspected lftomo configuration already has `energy_keV`, `pixel_size_um`
and `ssd_cm`. For parallel geometry these correspond to
`effective_energy_kev`, `geometry.pixel_m * 1e6`, and
`geometry.distance_m * 100`, respectively. These are unit conversions at the
consumer boundary, not additional values stored in this artifact.

lftomo #319 will consume the table's slope, affine intercept and envelope.
Its final new configuration keys are not implemented yet. In particular,
current `phase.paganin_filter.params.alpha` is the existing delta/beta filter parameter,
**not** this table's affine intercept; do not copy `line_fit.alpha` into it.
For a through-origin line, the existing denominator corresponds to a filter
parameter of `4*pi*line_fit.slope/wavelength_m`, using the consumer's wavelength
convention and converting its wavelength from centimetres to metres. #319 must
pin its deliberately duplicated denominator to the
reference on a shared fixture while retaining existing default behavior.

lftomo #320 will own the cone conversion from the three native geometry
inputs. lftomo #322 will consume the ordered μ/δ arrays at effective energy;
its material and classification configuration keys are also future work.
The supplied `single_distance_multimaterial` refits those arrays through the
origin and accepts optional material weights. A composition export may instead
have an affine fit. #322 must make that consumer fit policy explicit and retain
any non-default weights from `calibration.parameters.weights`; it must not
silently treat an affine export fit as the reference loop's through-origin fit.
This contract establishes plain data coupling without importing either
repository into the other and does not claim those consumers already exist.
