# Pressure diagnostics

The diagnostics implement the October 2026 guidance from Nami and Xu.
They stream one saved timestep at a time from `BOUT.dmp.bp`. No full time
series or interpolated visualization mesh is loaded for the reductions.

## Pressure definitions and units

Saved `P` is the perturbation relative to fixed `P0`, including the evolving
axisymmetric perturbation. Use:

```python
pbar = Bbar**2 / (2 * mu0)                 # Pa
p_total = pbar * (P0[..., None] + P)       # Pa
p_fluct = pbar * (P - P.mean(axis=-1, keepdims=True))
```

Plots and pressure profiles use kPa. There is no snapshot-maximum
normalization. Optional initial-pedestal normalization is deferred until the
initial pedestal reference is defined consistently across cases.

`Psi` is an evolving model field, not the equilibrium radial coordinate.
Radial profiles use `psixy`, `psi_axis`, and `psi_bndry` from the matching grid:

```python
psi_n = (psixy - psi_axis) / (psi_bndry - psi_axis)
```

## Toroidal spectra and integration

Fourier coefficients are `rfft(p_fluct_kpa) / N_unique`. For a one-sided power
spectrum, double all positive interior-bin powers, but not an even-length
Nyquist bin. The sum equals the toroidal fluctuation variance. The axisymmetric
bin is explicitly excluded after mean subtraction. Physical toroidal mode
numbers are `n = m * zperiod`.

Integrate each one-sided bin over the full valid native radial and poloidal
domain, including core, SOL, and private flux regions:

```python
w = abs(J * dx * dy) * Lbar**3
S_n = (bin_power_kpa2 * w[..., None]).sum(axis=(0, 1))
f_n = S_n / S_n[1:].sum()
```

This uses the reduced-MHD native coordinate normalization: radial spacing is
in `psi/(Bbar*Lbar**2)` and the normalized Jacobian is `hthe/Bp`.
`Lbar**3` restores physical volume. These are native-coordinate quadrature
weights, not a saved Jacobian relabeled as a `psi_n` Jacobian. Their integral
is equivalent to the coordinate-transformed integral with its transformed
Jacobian. Spectral reduction does not need resampling onto uniform `psi_n`.

`S_n` has units kPa² m³ per radian of toroidal angle. Multiplying its mode sum
by `2*pi` gives the full-torus pressure-variance integral. This diagnostic is
pressure fluctuation intensity, not thermodynamic energy or an ELM energy loss.

The integration domain excludes radial guard cells, poloidal guard cells,
reserved zero padding, and invalid static metric cells. Each native location
is counted once. The reader supports physical-only y arrays, outer guards,
double-null internal guards, and the contiguous gathered array with trailing
padding found in the laptop data. An optional duplicated closing toroidal
sample is checked and omitted from analysis.

Zero-fluctuation snapshots have zero modal fractions and undefined dominant
mode, centroid, and width. A `has_fluctuations` flag distinguishes these
snapshots. Pressure spectra are implemented; physical potential spectra await
the model's potential normalization.

## Grid pairing and radial profiles

Before reporting `psi_n` profiles or using the new scientific renderer modes,
the code checks grid/dump shapes and topology and compares `Bxy`, `dx`, and
`zShift` under standard `elm_pb` conversion assumptions. These assumptions
have not been verified against the exact model producing the laptop run.
The relative L2 tolerance is
`1e-4`. A missing or incompatible grid never supplies silently accepted radial
coordinates. `--grid-path` selects another matching BP grid without replacing
the existing data. When the original grid provenance is known, the renderer's
`--grid-confirmed` option accepts it after structural checks. Numerical
differences are still recorded in `grid_audit.json` and the frame manifest;
confirmation is not relabeled as passing the numerical comparisons.

For verified grids, profiles are:

- Outboard-midplane RMS: interpolate toroidally averaged variance at the
  outermost geometry crossing of `Z = --omp-z`, then take the square root.
  The default midplane is `Z=0` m. Branch boundaries are not interpolated across.
- Closed-surface RMS: average toroidal variance over the closed-core poloidal
  branches with weights `abs(J*dy)`, then take the square root.
- SOL and private-region RMS: separate weighted open-field poloidal averages.
  These are not labeled closed flux-surface averages.

The default analysis writes spectra even if grid pairing fails. Its metadata
reports profiles as unavailable. `--require-profiles` instead fails before
writing analysis products.

The laptop dataset inspected during implementation has 1,152 saved steps,
64 unique toroidal samples, `zperiod=5`, and four trailing padded y columns.
Its current `grid.bp` differs from the dump under the assumed `Bxy`, `dx`, and
`zShift` conversions. The user confirmed that this is the original input grid.
This does not establish whether the differences are expected preprocessing;
it does establish input-grid provenance for rendering. Local spectra use the
dump's own metrics. The standalone radial analysis retains its stricter
numerical check while the producing model's transformations remain unresolved.

## Running the analysis

Use an environment containing the existing `adios2`, `numpy`, and `matplotlib`
packages. The laptop environment is `/Users/dpn/venv/bin/python`.

```bash
python -B filament_diagnostics.py \
  --input /path/to/run --audit-only

python -B filament_diagnostics.py \
  --input /path/to/run --output /path/to/diagnostics

python -B filament_diagnostics.py \
  --input /path/to/run --grid-path /path/to/matching/grid.bp \
  --output /path/to/diagnostics --require-profiles
```

The defaults process all saved steps except the final 10. `--stop` is
exclusive and bypasses that trailing cutoff. `--start`, `--stride`, and
`--trailing-steps` can select another window. Absolute saved time and, when
`Tbar` is available, physical time in seconds are both retained.

Products are `diagnostics.npz`, `summary.csv`, `metadata.json`, and
`spectra.png`. A verified grid also produces RMS profiles and `profiles.png`.
Use `--no-plots` for reductions only. Existing products require a different
output directory or explicit `--overwrite`.

The NPZ contains mode numbers, absolute modal power, modal fractions, selected
steps/times, the valid-domain mask, and physical integration weights. With a
verified grid it also contains the four RMS profiles and their `psi_n`
coordinates. The CSV reports global RMS, dominant mode, centroid, width, and
the Parseval error for every selected step, as well as the mean total pressure
and the RMS axisymmetric perturbation. Metadata records all definitions,
units, geometry checks, masks, and selected steps.

## Signed filament rendering

Existing rendering defaults still contour saved `P` in code units. For the new
fluctuation visualization use a positive absolute threshold, for example:

```bash
python -B render_elmo_frames.py \
  --input /path/to/run --output /path/to/new/pressure_frames \
  --pressure-quantity fluctuation --iso-kpa 0.5 \
  --color-field p_fluct_kpa --color-limits -5 5 \
  --start 0 --stop 20
```

Add `--grid-confirmed` when the input grid's provenance has been confirmed and
only the model-dependent numerical comparisons fail.

This contours `P - mean_z(P)` at +0.5 kPa in red and -0.5 kPa in blue.
Empty surfaces are valid, including an initially zero perturbation. Both CLI
entry points support these options, `--grid-path`, and `--iso-value` for an
absolute code-unit threshold. `--pressure-quantity total` selects `P0 + P`.

For comparisons, use the same kPa threshold, color limits, and camera/domain
settings for every case. Fractions of separate case ranges remain available
in the legacy raw-P mode, but are not common physical amplitude thresholds.
The manifest records pressure quantity, units, pressure scale, positive and
negative levels, and cell counts. The parallel-gradient proxy remains in its
existing code units; it is not converted or relabeled as physical heat flux.
When `Tbar` is available, scientific frames show time in microseconds and the
manifest retains both code time and physical time in seconds.
Scientific rendering requires a new output directory or explicit overwrite,
so older raw-P frames cannot be silently reused as fluctuation frames.

## Transport status and verification

`pressure_fluxes(p_total_pa, v_r_ms)` implements the toroidal mean total
pressure flux and the covariance of pressure and velocity fluctuations, given
a separately validated physical radial velocity. Its output is Pa m/s.

The script does not yet derive radial velocity from `phi` or convert pressure
flux to enthalpy flux. The precise model revision, potential normalization,
metric-aware outward velocity expression, and enthalpy factor still need
verification. No radial transport product is claimed in the analysis output.

Run the analytic and BP fixture tests with:

```bash
python -B -m unittest -v test_filament_diagnostics
```

They check known Fourier modes, odd/even FFT lengths, Nyquist handling,
Parseval equality, variance-first RMS, mean/fluctuation flux decomposition,
guard/padding layouts, a closing toroidal sample, grid rejection, physical
normalization, zero-signal handling, and signed contour construction.
