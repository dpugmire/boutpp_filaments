#!/usr/bin/env python3
"""Streaming pressure spectra and radial RMS diagnostics for BOUT++ BP output.

Definitions follow the Nami/Xu analysis guidance of October 2026. Reductions
use native, unique samples, before any visualization interpolation. See
DIAGNOSTICS.md for units, domain masks, and the geometry validation contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import adios2
import numpy as np


MU0 = 4.0e-7 * np.pi


def pressure_scale(bbar: float) -> float:
    """Return the pressure normalization in Pa."""
    if not np.isfinite(bbar) or bbar <= 0.0:
        raise ValueError("Bbar must be finite and positive")
    return bbar * bbar / (2.0 * MU0)


def pressure_fluctuation(pressure: np.ndarray) -> np.ndarray:
    return pressure - np.mean(pressure, axis=-1, keepdims=True)


def one_sided_power(fluctuation: np.ndarray) -> np.ndarray:
    """Return variance-preserving bin powers; the last axis is toroidal."""
    count = fluctuation.shape[-1]
    if count < 2:
        raise ValueError("At least two unique toroidal samples are required")
    coefficients = np.fft.rfft(fluctuation, axis=-1) / count
    power = np.abs(coefficients) ** 2
    power[..., 1 : -1 if count % 2 == 0 else None] *= 2.0
    return power


def pressure_fluxes(
    total_pressure_pa: np.ndarray, radial_velocity_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Toroidal mean total and fluctuation pressure fluxes in Pa m/s.

    The caller must supply a validated physical radial velocity, positive
    outward. This does not derive velocity from phi or convert to enthalpy.
    """
    if total_pressure_pa.shape != radial_velocity_ms.shape:
        raise ValueError("Pressure and radial velocity must have the same shape")
    total = np.mean(total_pressure_pa * radial_velocity_ms, axis=-1)
    fluctuation = np.mean(
        pressure_fluctuation(total_pressure_pa)
        * pressure_fluctuation(radial_velocity_ms),
        axis=-1,
    )
    return total, fluctuation


def physical_y_indices(
    raw_ny: int, ny: int, myg: int, ny_inner: int, metric: np.ndarray
) -> tuple[np.ndarray, str]:
    """Identify physical y rows, including double-null guards or padding."""
    if raw_ny == ny:
        return np.arange(ny), "physical rows only"
    # Some gathered dumps reserve double-null space but populate a contiguous
    # ny+2*MYG array, leaving the last 2*MYG columns zero. Validate the padding
    # using a static metric, never by testing whether a perturbation is zero.
    populated_ny = ny + 2 * myg
    if raw_ny > populated_ny and np.all(metric[:, populated_ny:] == 0.0):
        return np.arange(myg, myg + ny), "outer guards and trailing zero padding"
    if raw_ny == populated_ny:
        return np.arange(myg, myg + ny), "outer guards"
    if raw_ny == ny + 4 * myg and 0 < ny_inner < ny:
        return (
            np.r_[myg : myg + ny_inner, ny_inner + 3 * myg : ny + 3 * myg],
            "outer and internal double-null guards",
        )
    raise ValueError(f"Unsupported y layout: raw_ny={raw_ny}, ny={ny}, MYG={myg}")


def weighted_rms(
    variance: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Average variance over y before taking its square root."""
    masked_weights = np.where(mask, weights, 0.0)
    denominator = masked_weights.sum(axis=1)
    mean_variance = np.divide(
        np.sum(variance * masked_weights, axis=1),
        denominator,
        out=np.full(variance.shape[0], np.nan),
        where=denominator > 0.0,
    )
    return np.sqrt(mean_variance)


class DiagnosticRun:
    """A BP reader and small static arrays; no VTK or full time-series load."""

    def __init__(
        self,
        run_dir: str | Path,
        grid_path: str | Path | None = None,
        *,
        grid_confirmed: bool = False,
    ):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.grid_confirmed = grid_confirmed
        self.grid_path = (
            Path(grid_path).expanduser().resolve()
            if grid_path is not None
            else self.run_dir / "grid.bp"
        )
        self.reader = adios2.FileReader(str(self.run_dir / "BOUT.dmp.bp"))
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __enter__(self) -> "DiagnosticRun":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def scalar(self, name: str, default=None):
        if name not in self.available:
            if default is not None:
                return default
            raise KeyError(f"Missing diagnostic variable: {name}")
        return np.asarray(self.reader.read(name)).reshape(-1)[0].item()

    def read(self, name: str, step: int | None = None) -> np.ndarray:
        if name not in self.available:
            raise KeyError(f"Missing diagnostic variable: {name}")
        count = int(self.available[name]["AvailableStepsCount"])
        if step is None or count == 1:
            values = self.reader.read(name)
        else:
            if not 0 <= step < count:
                raise IndexError(f"{name} has no saved step {step}")
            values = self.reader.read(name, step_selection=[step, 1])
        shape = tuple(int(v) for v in re.findall(r"\d+", self.available[name]["Shape"]))
        return np.asarray(values, dtype=np.float64).reshape(shape)

    def field(self, name: str, step: int | None = None) -> np.ndarray:
        values = self.read(name, step)
        if values.shape[:2] != (self.nx, self.raw_ny):
            raise ValueError(f"Unexpected {name} shape: {values.shape}")
        values = values[:, self.y_indices, ...]
        if values.ndim == 3:
            if values.shape[-1] != self.raw_nz:
                raise ValueError(f"{name} toroidal extent differs from P")
            if self.raw_nz != self.nz:
                if not np.allclose(values[..., 0], values[..., -1], atol=1e-14):
                    raise ValueError(f"{name} closing toroidal sample is not periodic")
                values = values[..., : self.nz]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} values")
        return values

    def _load(self) -> None:
        self.available = self.reader.available_variables()
        required = [
            "P",
            "P0",
            "Bbar",
            "Lbar",
            "J",
            "dx",
            "dy",
            "ny",
            "MYG",
            "MXG",
            "zperiod",
        ]
        missing = [n for n in required if n not in self.available]
        if missing:
            raise KeyError(f"Missing diagnostic variables: {missing}")
        shape = tuple(int(v) for v in re.findall(r"\d+", self.available["P"]["Shape"]))
        if len(shape) != 3:
            raise ValueError(f"Expected a 3-D P shape, got {shape}")
        self.nx, self.raw_ny, self.raw_nz = shape
        self.ny = int(self.scalar("ny"))
        self.myg = int(self.scalar("MYG"))
        self.mxg = int(self.scalar("MXG"))
        self.ny_inner = int(self.scalar("ny_inner", self.ny))
        if min(self.myg, self.mxg) < 0 or 2 * self.mxg >= self.nx:
            raise ValueError("Invalid guard-cell extents")
        self.y_indices, layout = physical_y_indices(
            self.raw_ny, self.ny, self.myg, self.ny_inner, self.read("J")
        )
        # A padded dump must have consistent padding in all integration metrics.
        if "padding" in layout:
            for name in ["dx", "dy", "P0"]:
                if np.any(self.read(name)[:, self.ny + 2 * self.myg :] != 0.0):
                    raise ValueError(f"Inconsistent trailing padding in {name}")
        self.nz = int(self.scalar("nz", self.raw_nz))
        if self.raw_nz not in (self.nz, self.nz + 1):
            raise ValueError("P toroidal extent does not match nz or nz+1")
        self.zperiod = int(self.scalar("zperiod"))
        if self.zperiod <= 0 or self.nz < 2:
            raise ValueError("Invalid toroidal sample count or zperiod")
        self.nsteps = int(self.available["P"]["AvailableStepsCount"])
        self.bbar = float(self.scalar("Bbar"))
        self.lbar = float(self.scalar("Lbar"))
        if not np.isfinite(self.lbar) or self.lbar <= 0.0:
            raise ValueError("Lbar must be finite and positive")
        self.pbar_pa = pressure_scale(self.bbar)
        self.tbar = float(self.scalar("Tbar", 0.0))
        self.p0 = self.field("P0")
        self.jacobian = self.field("J")
        self.dx = self.field("dx")
        self.dy = self.field("dy")
        self.valid = (self.dx > 0.0) & (self.dy > 0.0) & (self.jacobian != 0.0)
        if self.mxg:
            self.valid[: self.mxg] = False
            self.valid[-self.mxg :] = False
        if not np.any(self.valid):
            raise ValueError("No valid integration cells")
        # In the reduced-MHD normalization: x = psi/(Bbar*Lbar**2),
        # J = hthe/Bp in normalized units. Lbar**3 restores physical volume.
        self.weights = np.where(
            self.valid, np.abs(self.jacobian * self.dx * self.dy) * self.lbar**3, 0.0
        )
        self.surface_weights = np.abs(self.jacobian * self.dy)
        if "dz" in self.available:
            dz = self.field("dz")
            if not np.allclose(dz[self.valid] * self.nz, 2.0 * np.pi / self.zperiod):
                raise ValueError("Toroidal spacing does not match nz and zperiod")
        self.audit = {
            "input_dir": str(self.run_dir),
            "grid_path": str(self.grid_path),
            "raw_shape": list(shape),
            "physical_shape": [self.nx, self.ny, self.nz],
            "y_layout": layout,
            "physical_y_indices": self.y_indices.tolist(),
            "MXG": self.mxg,
            "valid_cells": int(self.valid.sum()),
            "excluded_physical_cells": int(self.valid.size - self.valid.sum()),
            "steps_available": self.nsteps,
            "Bbar_T": self.bbar,
            "Lbar_m": self.lbar,
            "Tbar_s": self.tbar or None,
            "pbar_Pa": self.pbar_pa,
            "zperiod": self.zperiod,
            "unique_toroidal_samples": self.nz,
            "grid_checks": [],
        }
        self._check_grid()

    def _check_grid(self) -> None:
        self.grid = {}
        checks = self.audit["grid_checks"]
        if not self.grid_path.exists():
            checks.append({"name": "grid exists", "passed": False})
        else:
            with adios2.FileReader(str(self.grid_path)) as reader:
                names = reader.available_variables()
                needed = [
                    "Rxy",
                    "Zxy",
                    "psixy",
                    "psi_axis",
                    "psi_bndry",
                    "Bxy",
                    "zShift",
                    "dx",
                ]
                missing = [name for name in needed if name not in names]
                checks.append(
                    {
                        "name": "grid variables",
                        "passed": not missing,
                        "missing": missing,
                    }
                )
                if not missing:
                    for name in needed + ["bmag", "rmag", "ShiftAngle"]:
                        if name in names:
                            self.grid[name] = np.asarray(
                                reader.read(name), dtype=np.float64
                            )
                    for name in ["Rxy", "Zxy", "psixy", "Bxy", "zShift", "dx"]:
                        values = self.grid[name]
                        checks.append(
                            {
                                "name": f"{name} shape",
                                "passed": values.shape == (self.nx, self.ny)
                                and bool(np.all(np.isfinite(values))),
                            }
                        )
                    for name in [
                        "nx",
                        "ny",
                        "ny_inner",
                        "ixseps1",
                        "ixseps2",
                        "jyseps1_1",
                        "jyseps1_2",
                        "jyseps2_1",
                        "jyseps2_2",
                    ]:
                        matches = (
                            name in names
                            and name in self.available
                            and np.asarray(reader.read(name)).size == 1
                            and np.asarray(reader.read(name)).item()
                            == self.scalar(name)
                        )
                        checks.append(
                            {"name": f"topology {name}", "passed": bool(matches)}
                        )
                    if all(c["passed"] for c in checks):
                        for dump, grid, scale in [
                            ("Bxy", "Bxy", self.bbar),
                            ("dx", "dx", self.bbar * self.lbar**2),
                            ("zShift", "zShift", 1.0),
                        ]:
                            if dump not in self.available:
                                checks.append(
                                    {
                                        "name": f"{dump} pairing",
                                        "passed": False,
                                        "reason": "dump variable missing",
                                    }
                                )
                                continue
                            expected = self.grid[grid][self.valid]
                            actual = self.field(dump)[self.valid] * scale
                            error = float(
                                np.linalg.norm(actual - expected)
                                / max(np.linalg.norm(expected), np.finfo(float).tiny)
                            )
                            checks.append(
                                {
                                    "name": f"{dump} pairing",
                                    "passed": error <= 1.0e-4,
                                    "relative_l2_error": error,
                                }
                            )
                    self.audit["grid_bmag_T"] = (
                        float(self.grid["bmag"]) if "bmag" in self.grid else None
                    )
                    self.audit["grid_rmag_m"] = (
                        float(self.grid["rmag"]) if "rmag" in self.grid else None
                    )
        self.geometry_verified = bool(checks) and all(c["passed"] for c in checks)
        structural_checks = [c for c in checks if not c["name"].endswith(" pairing")]
        self.grid_structure_valid = bool(structural_checks) and all(
            c["passed"] for c in structural_checks
        )
        self.geometry_accepted = self.grid_structure_valid and (
            self.geometry_verified or self.grid_confirmed
        )
        self.audit["geometry_verified"] = self.geometry_verified
        self.audit["grid_structure_valid"] = self.grid_structure_valid
        self.audit["grid_provenance_confirmed"] = self.grid_confirmed
        self.audit["geometry_accepted"] = self.geometry_accepted
        self.audit["geometry_acceptance_basis"] = (
            "numerical checks"
            if self.geometry_verified
            else "user-confirmed input-grid provenance"
            if self.geometry_accepted
            else "not accepted"
        )
        self.audit["numerical_check_assumptions"] = (
            "Standard elm_pb conversions; the exact producing model is not verified. "
            "Numerical differences alone do not establish a different input grid."
        )

    def require_geometry(self) -> None:
        if not self.geometry_accepted:
            failures = [c["name"] for c in self.audit["grid_checks"] if not c["passed"]]
            raise ValueError(
                "Grid/dump pairing is not verified: "
                + ", ".join(failures)
                + ". Check the grid structure and producing model conventions. "
                "The renderer accepts --grid-confirmed for a known original input grid."
            )

    def profile_geometry(self, omp_z: float = 0.0) -> dict:
        self.require_geometry()
        psi_axis = float(self.grid["psi_axis"])
        psi_sep = float(self.grid["psi_bndry"])
        if psi_sep == psi_axis:
            raise ValueError("psi_axis and psi_bndry must differ")
        psi_n = (self.grid["psixy"] - psi_axis) / (psi_sep - psi_axis)
        ixseps = int(self.scalar("ixseps1"))
        if ixseps != int(self.scalar("ixseps2")):
            raise ValueError("Radial profiles currently require ixseps1 == ixseps2")
        y = np.arange(self.ny)
        core_y = ((y > self.scalar("jyseps1_1")) & (y <= self.scalar("jyseps2_1"))) | (
            (y > self.scalar("jyseps1_2")) & (y <= self.scalar("jyseps2_2"))
        )
        inner = np.arange(self.nx)[:, None] < ixseps
        core = self.valid & inner & core_y[None, :] & (psi_n < 1.0)
        sol = self.valid & ~inner
        private = self.valid & inner & ~core_y[None, :]
        masks = {"closed": core, "sol": sol, "private": private}
        result = {"psi_n": psi_n, "masks": masks}
        for name, mask in masks.items():
            weights = np.where(mask, self.surface_weights, 0.0)
            result[f"psi_n_{name}"] = np.divide(
                np.sum(weights * psi_n, axis=1),
                weights.sum(axis=1),
                out=np.full(self.nx, np.nan),
                where=weights.sum(axis=1) > 0.0,
            )
        # Interpolate variance at Z=omp_z, choosing the outermost crossing.
        # Do not interpolate across disconnected divertor branches.
        left = np.full(self.nx, -1, dtype=int)
        fraction = np.zeros(self.nx)
        omp_psi = np.full(self.nx, np.nan)
        radius, height = self.grid["Rxy"], self.grid["Zxy"]
        for x in range(self.mxg, self.nx - self.mxg):
            candidates = []
            for j in range(self.ny - 1):
                if j + 1 == self.ny_inner or not (core_y[j] and core_y[j + 1]):
                    continue
                if not (self.valid[x, j] and self.valid[x, j + 1]):
                    continue
                z1, z2 = height[x, j : j + 2]
                if z1 == z2 or not min(z1, z2) <= omp_z <= max(z1, z2):
                    continue
                f = (omp_z - z1) / (z2 - z1)
                r = (1.0 - f) * radius[x, j] + f * radius[x, j + 1]
                candidates.append((r, j, f))
            if candidates:
                _, j, f = max(candidates)
                left[x], fraction[x] = j, f
                omp_psi[x] = (1.0 - f) * psi_n[x, j] + f * psi_n[x, j + 1]
        result.update(omp_left=left, omp_fraction=fraction, psi_n_omp=omp_psi)
        return result


def radial_profiles(variance: np.ndarray, run: DiagnosticRun, geometry: dict) -> dict:
    result = {
        f"rms_{name}_kpa": weighted_rms(variance, run.surface_weights, mask)
        for name, mask in geometry["masks"].items()
    }
    left, fraction = geometry["omp_left"], geometry["omp_fraction"]
    valid = left >= 0
    rows, columns = np.where(valid)[0], left[valid]
    omp_variance = np.full(run.nx, np.nan)
    omp_variance[valid] = (1.0 - fraction[valid]) * variance[rows, columns] + fraction[
        valid
    ] * variance[rows, columns + 1]
    result["rms_omp_kpa"] = np.sqrt(omp_variance)
    return result


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    grid_path=None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    trailing_steps: int = 10,
    require_profiles: bool = False,
    omp_z: float = 0.0,
    plots: bool = True,
    overwrite: bool = False,
) -> Path:
    if start < 0 or stride <= 0 or trailing_steps < 0:
        raise ValueError("Invalid timestep selection")
    output = Path(output_dir).expanduser().resolve()
    products = [
        "diagnostics.npz",
        "metadata.json",
        "summary.csv",
        "spectra.png",
        "profiles.png",
    ]
    if not overwrite and any((output / name).exists() for name in products):
        raise FileExistsError(
            "Diagnostic products exist; use a new output or --overwrite"
        )
    with DiagnosticRun(input_dir, grid_path) as run:
        last = run.nsteps - trailing_steps if stop is None else min(stop, run.nsteps)
        steps = np.arange(start, last, stride)
        if not len(steps):
            raise ValueError("Empty timestep selection")
        if require_profiles:
            run.require_geometry()
        geometry = run.profile_geometry(omp_z) if run.geometry_verified else None
        if overwrite:
            obsolete = []
            if not plots:
                obsolete.extend(
                    name
                    for name in ["spectra.png", "profiles.png"]
                    if (output / name).exists()
                )
            elif geometry is None and (output / "profiles.png").exists():
                obsolete.append("profiles.png")
            if obsolete:
                raise FileExistsError(
                    "This selection would leave stale plots: "
                    + ", ".join(obsolete)
                    + ". Use a new output directory."
                )
        if geometry is None:
            print(
                "Grid pairing failed: spectra will be saved; psi_N profiles are unavailable.",
                flush=True,
            )
        modes = np.arange(run.nz // 2 + 1) * run.zperiod
        powers, profiles, rows = [], [], []
        times = []
        max_parseval_relative = 0.0
        for index, step in enumerate(steps):
            pressure = run.field("P", int(step)) * (run.pbar_pa / 1000.0)
            toroidal_mean = np.mean(pressure, axis=-1)
            fluctuation = pressure_fluctuation(pressure)
            variance = np.mean(fluctuation**2, axis=-1)
            local_power = one_sided_power(fluctuation)
            # Axisymmetric power is excluded explicitly; mean-subtraction
            # leaves only roundoff in this bin.
            local_power[..., 0] = 0.0
            s_n = np.sum(local_power * run.weights[..., None], axis=(0, 1))
            total = float(s_n.sum())
            variance_integral = float(np.sum(variance * run.weights))
            parseval = abs(total - variance_integral) / max(
                variance_integral, np.finfo(float).tiny
            )
            if parseval > 1.0e-10:
                raise FloatingPointError(
                    f"Parseval check failed at step {step}: {parseval}"
                )
            max_parseval_relative = max(max_parseval_relative, parseval)
            fractions = s_n / total if total > 0.0 else np.zeros_like(s_n)
            centroid = float(np.sum(modes * fractions)) if total else np.nan
            width = (
                float(np.sqrt(np.sum((modes - centroid) ** 2 * fractions)))
                if total
                else np.nan
            )
            dominant = int(modes[np.argmax(s_n[1:]) + 1]) if total else np.nan
            time_name = "t_array" if "t_array" in run.available else "t"
            time = (
                float(run.read(time_name, int(step)))
                if time_name in run.available
                else np.nan
            )
            times.append(time)
            powers.append(s_n)
            if geometry is not None:
                profiles.append(radial_profiles(variance, run, geometry))
            rows.append(
                {
                    "step": int(step),
                    "time_code": time,
                    "time_s": time * run.tbar if run.tbar else np.nan,
                    "total_modal_power_kpa2_m3_per_rad": total,
                    "global_rms_kpa": float(
                        np.sqrt(variance_integral / run.weights.sum())
                    ),
                    "mean_total_pressure_kpa": float(
                        np.sum(
                            (toroidal_mean + run.p0 * run.pbar_pa / 1000.0)
                            * run.weights
                        )
                        / run.weights.sum()
                    ),
                    "axisymmetric_perturbation_rms_kpa": float(
                        np.sqrt(
                            np.sum(toroidal_mean**2 * run.weights) / run.weights.sum()
                        )
                    ),
                    "dominant_n": dominant,
                    "centroid_n": centroid,
                    "width_n": width,
                    "has_fluctuations": bool(total > 0.0),
                    "parseval_relative_error": parseval,
                }
            )
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(steps):
                print(
                    f"Analyzed {index + 1}/{len(steps)}: step={step}, RMS={rows[-1]['global_rms_kpa']:.6g} kPa",
                    flush=True,
                )
        powers = np.asarray(powers)
        totals = powers.sum(axis=1, keepdims=True)
        data = {
            "step": steps,
            "time_code": np.asarray(times),
            "n": modes,
            "modal_power_kpa2_m3_per_rad": powers,
            "modal_fraction": np.divide(
                powers, totals, out=np.zeros_like(powers), where=totals > 0.0
            ),
            "valid_xy": run.valid,
            "physical_y_indices": run.y_indices,
            "weights_m3_per_rad": run.weights,
        }
        if run.tbar:
            data["time_s"] = np.asarray(times) * run.tbar
        if geometry is not None:
            for name in ["omp", "closed", "sol", "private"]:
                data[f"psi_n_{name}"] = geometry[f"psi_n_{name}"]
                data[f"rms_{name}_kpa"] = np.asarray(
                    [p[f"rms_{name}_kpa"] for p in profiles]
                )
        metadata = dict(run.audit)
        metadata.update(
            pressure_definition="p_fluct = Bbar**2/(2*mu0) * (P - mean_z(P)); displayed in kPa",
            total_pressure_definition="p_total = Bbar**2/(2*mu0) * (P0 + P)",
            power_definition="S_n = sum_xy(one_sided_bin_power_kPa2 * abs(J*dx*dy) * Lbar**3)",
            power_units="kPa^2 m^3 per radian of toroidal angle; multiply sum_n(S_n) by 2*pi for full-torus variance integral",
            jacobian_coordinates="Native normalized BOUT coordinates, including cell spacings; not J relabeled as a psi_N Jacobian",
            normalization="FFT/N_unique; double positive interior bins; do not double Nyquist",
            zero_signal="modal fractions are zero and dominant/centroid/width are undefined",
            profile_status="available"
            if geometry is not None
            else "unavailable: grid/dump pairing failed",
            surface_definition="sqrt(weighted_y(mean_z(p_fluct**2))), weights=abs(J*dy); closed topology only",
            open_profiles="SOL/private profiles are separate open-field poloidal averages, not closed flux-surface averages",
            omp_z_m=omp_z,
            transport_status="not calculated: model-specific radial velocity and enthalpy conversion unverified",
            selected_steps=steps.tolist(),
            trailing_steps=trailing_steps,
            max_parseval_relative_error=max_parseval_relative,
        )
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "diagnostics.npz", **data)
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if plots:
            plot_diagnostics(data, output)
    return output / "diagnostics.npz"


def plot_diagnostics(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    time = data.get("time_s", data["time_code"])
    time_label = "Time (s)" if "time_s" in data else "Simulation time"
    if not np.all(np.isfinite(time)):
        time, time_label = data["step"], "Saved step"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    power = data["modal_power_kpa2_m3_per_rad"][:, 1:]
    positive = power[power > 0.0]
    limits = (
        (
            max(float(positive.min()), float(positive.max()) * 1e-12),
            float(positive.max()),
        )
        if positive.size
        else (1e-12, 1.0)
    )
    if limits[0] == limits[1]:
        limits = (limits[0] * 0.1, limits[1] * 10.0)
    for axis, values, title, norm in [
        (
            axes[0],
            np.ma.masked_less_equal(power.T, 0),
            "Absolute modal power (kPa² m³/rad)",
            LogNorm(*limits),
        ),
        (
            axes[1],
            data["modal_fraction"][:, 1:].T,
            "Modal fraction",
            Normalize(0.0, 1.0),
        ),
    ]:
        artist = axis.pcolormesh(
            time, data["n"][1:], values, shading="nearest", norm=norm, cmap="viridis"
        )
        axis.set(xlabel=time_label, ylabel="Toroidal mode n", title=title)
        fig.colorbar(artist, ax=axis)
    fig.savefig(output / "spectra.png", dpi=160)
    plt.close(fig)
    if "rms_omp_kpa" in data:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        selected = np.unique(
            np.linspace(0, len(time) - 1, min(4, len(time)), dtype=int)
        )
        for axis, name, title in [
            (axes[0], "omp", "Outboard midplane"),
            (axes[1], "closed", "Closed flux surfaces"),
        ]:
            x = data[f"psi_n_{name}"]
            for index in selected:
                unit = " s" if "time_s" in data else ""
                axis.plot(
                    x,
                    data[f"rms_{name}_kpa"][index],
                    label=f"t={time[index]:.3g}{unit}",
                )
            axis.axvline(1.0, color="gray", linestyle=":")
            axis.set(
                xlabel="Normalized poloidal flux ψN",
                ylabel="Pressure RMS (kPa)",
                title=title,
            )
            axis.legend()
        fig.savefig(output / "profiles.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--grid-path", type=Path, help="Matching grid.bp; defaults to <input>/grid.bp"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, help="Exclusive saved-step stop")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--trailing-steps", type=int, default=10)
    parser.add_argument(
        "--require-profiles",
        action="store_true",
        help="Fail if the grid/dump pairing is unverified",
    )
    parser.add_argument(
        "--omp-z",
        type=float,
        default=0.0,
        help="Midplane height in meters (default: 0)",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        with DiagnosticRun(args.input, args.grid_path) as run:
            print(json.dumps(run.audit, indent=2, allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required unless --audit-only is used")
    result = analyze(
        args.input,
        args.output,
        grid_path=args.grid_path,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        trailing_steps=args.trailing_steps,
        require_profiles=args.require_profiles,
        omp_z=args.omp_z,
        plots=not args.no_plots,
        overwrite=args.overwrite,
    )
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
