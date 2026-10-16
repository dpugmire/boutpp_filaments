# BOUT++ Filament Visualization and Pressure Diagnostics

This repository reads BOUT++ ADIOS BP output to render three-dimensional
filament frames and compute pressure fluctuation spectra and radial RMS
profiles. It also contains a NetCDF-to-BP grid converter.

For complete diagnostic definitions, units, assumptions, and validation
behavior, see [DIAGNOSTICS.md](DIAGNOSTICS.md).

## Input data

Each simulation case is a directory containing the time-dependent output and
its static grid:

```text
CASE_DIR/
├── BOUT.dmp.bp/
└── grid.bp/
```

These BP datasets are directories. Pass the case directory to the scripts;
they locate both datasets automatically. The diagnostic reader expects the
pressure, normalization, integration metrics, and topology variables described
in `DIAGNOSTICS.md`. Rendering also needs the static geometry in `grid.bp` and
the selected background color field, which defaults to `phi`.

The renderer requires `P`, `dy`, `g_22`, `diffusion_par`, `ny`, `MYG`,
`zperiod`, and the two-X-point topology variables in `BOUT.dmp.bp`; it also
needs `Rxy`, `Zxy`, `zShift`, and `ShiftAngle` in `grid.bp`. The selected
background field must be present in the BP output. The current renderer
supports the two-X-point topology used by these ELMO runs and requires
`ixseps1 == ixseps2`.

## Environment

Use Python 3.10 or newer with these packages:

- Pressure diagnostics: `adios2`, `numpy`, and `matplotlib`
- Frame rendering: `adios2`, `numpy`, `pyvista`, and VTK
- NetCDF grid conversion: additionally `scipy`

No dependency lock file is currently provided. On NERSC, activate an
environment on a persistent filesystem that contains the needed packages.
Rendering is off-screen; VTK must support EGL, OSMesa, or a configured virtual
X server. Check imports and render a short smoke test before submitting a long
job.

For workflow provenance, record the source revision with:

```bash
git rev-parse HEAD
```

## Pressure diagnostics

Run a fast input and grid audit:

```bash
python -B filament_diagnostics.py \
  --input "$CASE_DIR" \
  --audit-only
```

Compute spectra and summary data:

```bash
python -B filament_diagnostics.py \
  --input "$CASE_DIR" \
  --output "$DIAGNOSTICS_DIR"
```

The default analyzes all saved steps except the final 10. Set
`--trailing-steps 0` to include every step. An explicit `--stop` is exclusive
and takes precedence over the trailing-step cutoff. Use `--start`, `--stop`,
and `--stride` to select another interval.

The analysis streams one timestep at a time and writes:

- `diagnostics.npz`: modal powers and fractions, times, valid-domain mask,
  integration weights, and any available profiles
- `summary.csv`: global RMS, mean total pressure, axisymmetric perturbation
  RMS, dominant toroidal mode, spectral centroid/width, and Parseval error
- `metadata.json`: definitions, units, selected steps, domain mask, and grid
  checks
- `spectra.png`: absolute modal power and modal fractions over time
- `profiles.png`: RMS profiles, only when the grid pairing is verified

If grid checks fail, spectra are still written using metrics from the dump,
but `psi_N` profiles are unavailable. `--require-profiles` makes the run fail
instead of producing spectra-only results. Do not use that option until the
grid and dump are verified as a pair.

### Definitions used

Saved `P` is the pressure perturbation relative to fixed `P0`. The total
pressure and nonaxisymmetric fluctuation are:

```text
p_total = pbar * (P0 + P)
p_fluct = pbar * (P - mean_z(P))
pbar    = Bbar^2 / (2 mu0)
```

Pressure is reported in kPa without per-snapshot maximum normalization. The
toroidal FFT uses `rfft(p_fluct) / N_unique`; one-sided interior-bin power is
doubled, the Nyquist bin is not, and the mode number is `n = m * zperiod`.
Modal power is integrated across the valid radial and poloidal domain using
the native BOUT coordinate weights. `S_n` is in kPa² m³ per radian; it is
pressure fluctuation intensity, not thermodynamic energy or ELM energy loss.

Radial profiles use equilibrium `psixy`, `psi_axis`, and `psi_bndry` to form
`psi_N`. The RMS is computed by averaging toroidal variance over the surface
before taking the square root. OMP and closed-surface profiles require a
verified grid. Closed-surface behavior uses the closed-core poloidal branches;
SOL and private-region values are separate open-field averages, not closed
flux-surface averages. The current code does not calculate physical potential
spectra, derive physical radial velocity from `phi`, or report pressure
transport and enthalpy flux. `U` is not assumed to be the radial velocity.

The grid audit compares shapes and topology, then compares `Bxy`, `dx`, and
`zShift` using standard `elm_pb` conversion assumptions (relative L2 tolerance
`1e-4`). These assumptions have not been confirmed for every model revision.
If those numerical comparisons fail, profiles remain unavailable unless the
grid is accepted through a separate user provenance confirmation in the
renderer. That confirmation cannot override a shape or topology mismatch and
does not mark the numerical checks as passed. The spectra use the dump's own
integration metrics and do not require resampling onto `psi_N`.

## Filament frame rendering

Start with a two-frame smoke test. `--stop` is exclusive, so this renders
steps 0 and 1:

```bash
python -B render_elmo_frames.py \
  --input "$CASE_DIR" \
  --output "$FRAME_OUTPUT" \
  --start 0 \
  --stop 2 \
  --stride 1
```

Inspect the PNGs and `manifest.csv` before scaling up. The wrapper writes frames
under `$FRAME_OUTPUT/elmo/`. Existing frames are skipped unless `--overwrite`
is supplied.

The legacy defaults contour saved `P` in code units, color the background by
`phi`, and use a per-timestep isovalue at 50% of the `P` range. When `--stop`
is omitted, the final 10 saved steps are skipped. Use `--trailing-steps 0` to
render through the last saved step.

For the signed physical-pressure visualization, use a fixed common threshold:

```bash
python -B render_elmo_frames.py \
  --input "$CASE_DIR" \
  --output "$FRAME_OUTPUT" \
  --pressure-quantity fluctuation \
  --iso-kpa 0.5 \
  --color-field p_fluct_kpa \
  --color-limits -5 5 \
  --start 0 \
  --stride 1 \
  --trailing-steps 10
```

This contours `P - mean_z(P)` at +0.5 kPa (red) and -0.5 kPa (blue), with the
background colored by the same fluctuation on fixed limits. Set
`--trailing-steps 0` if all saved steps are wanted. For comparisons across
cases, keep the pressure thresholds, color limits, camera, and domain settings
consistent.

If provenance confirms that the input grid is the original grid for this run,
add `--grid-confirmed` when only the model-dependent numerical comparisons
fail. Structural and topology checks still apply, and numerical differences
remain recorded in `grid_audit.json` and the frame manifest. Do not use this
option for an unverified grid.

For several named runs, `filament_renderer.py` accepts repeated
`--input name=CASE_DIR` options and processes the cases sequentially. Its
default start and stride differ from the wrapper, so specify them explicitly
for workflow runs. Useful options include `--iso-value` for a fixed threshold
in code units, `--iso-range START STOP` for a shared pressure range, and
`--fail-fast` to stop at the first frame error. The general script also
supports `--write-grid-vtk` and `--write-merged-grid-vtk` for volume-grid
exports.

Each input gets its own output subdirectory with numbered
`filament.NNNNN.png` frames, `manifest.csv`, and `background.vtk`. The
manifest records timesteps, simulation times, threshold and field ranges, cell
counts, per-frame status, and errors. The generic renderer continues after
frame failures unless `--fail-fast` is used. Existing frames are skipped
unless `--overwrite` is specified.

## NERSC workflow outline

Use one job per case or a Slurm job array for independent cases. First run the
two-frame smoke test above in an interactive allocation. For a full case, a
batch script can run analysis and rendering sequentially:

```bash
#!/bin/bash
#SBATCH -A <allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -J filament-analysis
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

cd /path/to/boutpp_filaments
source "$SCRATCH/venvs/boutpp-filaments/bin/activate"

export CASE_DIR=/path/to/BOUT/case
export DIAGNOSTICS_DIR="$SCRATCH/filament-diagnostics/case-name"
export FRAME_OUTPUT="$SCRATCH/filament-frames/case-name"
export PYVISTA_OFF_SCREEN=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$DIAGNOSTICS_DIR" "$FRAME_OUTPUT"

srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" \
  python -B filament_diagnostics.py \
    --input "$CASE_DIR" \
    --output "$DIAGNOSTICS_DIR" \
    --trailing-steps 10

srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" \
  python -B render_elmo_frames.py \
    --input "$CASE_DIR" \
    --output "$FRAME_OUTPUT" \
    --pressure-quantity fluctuation \
    --iso-kpa 0.5 \
    --color-field p_fluct_kpa \
    --color-limits -5 5 \
    --start 0 \
    --stride 1 \
    --trailing-steps 10
```

Replace the allocation, paths, environment, and wall time. Add
`--grid-confirmed` to the renderer command only when the exact grid provenance
is established. Submit with `sbatch`. Check the Slurm log, `manifest.csv`, and
`metadata.json` before treating the workflow as successful. If VTK fails to
initialize headlessly, use a supported EGL/OSMesa environment or test with
`xvfb-run` if available.

The renderer is serial across cases and timesteps. For independent cases,
prefer separate jobs or a Slurm job array. CPU allocation in the example is a
starting point, not a measured optimum. Full volume-grid exports are much
larger than the boundary mesh used for rendering: a representative ELMO
multiblock export was about 1.0 GB and a point-welded VTU was about 1.69 GB.
Plan memory and filesystem space before requesting these exports. The merged
VTU welds coincident points (including the toroidal seam); the seven-block
VTM/VTS export retains duplicate points at block interfaces and the seam.

## Grid conversion

To convert a NetCDF BOUT++ grid to a static one-step ADIOS BP dataset:

```bash
python -B convert_grid_nc_to_bp.py /path/to/grid.nc /path/to/grid.bp
```

The converter reads the new BP dataset back and compares its variables with
the NetCDF source. It requires `scipy`. `--no-verify` skips the read-back
comparison. `--overwrite` deletes and replaces the specified output dataset;
check the destination path before using it.

## Tests and optional movie encoding

Run the analytic and BP fixture tests from the repository root:

```bash
python -B -m unittest -v test_filament_diagnostics
```

If `ffmpeg` is available, encode completed frames from their directory:

```bash
ffmpeg -framerate 30 \
  -pattern_type glob -i 'filament.*.png' \
  -vf 'scale=960:-2' \
  -c:v libx264 -crf 28 -preset medium -pix_fmt yuv420p \
  filament.mp4
```

The NetCDF-to-BP grid converter is `convert_grid_nc_to_bp.py`; run it with
`--help` for all options. Optional full-grid VTK exports are available through
the `--write-grid-vtk` and `--write-merged-grid-vtk` renderer flags described
above.
