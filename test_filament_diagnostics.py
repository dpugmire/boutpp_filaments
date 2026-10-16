"""Analytic FFT checks and small ADIOS fixtures for pressure diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import adios2
import numpy as np

from filament_diagnostics import (
    DiagnosticRun,
    analyze,
    one_sided_power,
    physical_y_indices,
    pressure_fluctuation,
    pressure_fluxes,
    pressure_scale,
    radial_profiles,
    weighted_rms,
)


def write_fixture(root: Path, layout="padded", closing_plane=False) -> np.ndarray:
    nx, ny, nz, myg, ny_inner = 8, 12, 64, 1, 6
    raw_ny = ny + 4 * myg
    y_indices = np.arange(1, 13) if layout == "padded" else np.r_[1:7, 9:15]
    shape = (nx, raw_ny)
    theta = 2.0 * np.pi * (np.arange(ny) - 3.5) / ny
    x = np.arange(nx)[:, None]
    radius = (1.5 + 0.3 * np.cos(theta)[None, :]) * (1.0 + 0.02 * x)
    height = np.broadcast_to(0.4 * np.sin(theta), (nx, ny)).copy()
    psi_n = np.broadcast_to(np.linspace(0.7, 1.2, nx)[:, None], (nx, ny)).copy()
    metric = np.broadcast_to(1.0 + 0.2 * np.cos(theta), (nx, ny)).copy()
    amplitude = 1e-4 * (1.0 + 0.1 * x + 0.05 * np.arange(ny)[None, :])
    amplitude[[0, -1], :] = 10.0  # Huge guard signals must not enter reductions.

    def raw(values):
        a = np.ones(shape) if layout == "double-null" else np.zeros(shape)
        a[:, y_indices] = values
        return a

    topology = {
        "ixseps1": 5,
        "ixseps2": 5,
        "ny_inner": ny_inner,
        "jyseps1_1": 1,
        "jyseps2_1": 4,
        "jyseps1_2": 7,
        "jyseps2_2": 10,
    }
    static = {
        "P0": raw(np.full((nx, ny), 0.03)),
        "J": raw(metric),
        "dx": raw(np.full((nx, ny), 0.01)),
        "dy": raw(np.full((nx, ny), 0.1)),
        "dz": raw(np.full((nx, ny), 2.0 * np.pi / (nz * 5))),
        "Bxy": raw(np.ones((nx, ny))),
        "zShift": raw(np.zeros((nx, ny))),
        "g_22": raw(np.ones((nx, ny))),
        "Bbar": 2.0,
        "Lbar": 1.0,
        "Tbar": 1e-6,
        "diffusion_par": 0.01,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "MXG": 1,
        "MYG": myg,
        "zperiod": 5,
        **topology,
    }
    grid = {
        "Rxy": radius,
        "Zxy": height,
        "psixy": psi_n,
        "psi_axis": 0.0,
        "psi_bndry": 1.0,
        "Bxy": np.full((nx, ny), 2.0),
        "zShift": np.zeros((nx, ny)),
        "dx": np.full((nx, ny), 0.02),
        "dy": np.full((nx, ny), 0.1),
        "ShiftAngle": np.zeros(nx),
        "bmag": 2.0,
        "rmag": 1.0,
        "nx": nx,
        "ny": ny,
        **topology,
    }
    with adios2.Stream(str(root / "grid.bp"), "w") as stream:
        stream.begin_step()
        for name, values in grid.items():
            stream.write(name, np.asarray(values))
        stream.end_step()
    with adios2.Stream(str(root / "BOUT.dmp.bp"), "w") as stream:
        for step in range(3):
            stream.begin_step()
            if step == 0:
                for name, values in static.items():
                    stream.write(name, np.asarray(values))
            fluctuation = amplitude[..., None] * np.cos(
                3 * 2 * np.pi * np.arange(nz) / nz
            )
            p = np.zeros((nx, raw_ny, nz))
            p[:, y_indices, :] = 0.02 * step + step * fluctuation
            if closing_plane:
                p = np.concatenate((p, p[..., :1]), axis=-1)
            stream.write("P", p)
            stream.write("phi", p * 0.1)
            stream.write("t_array", np.asarray(float(step)))
            stream.end_step()
    return amplitude


class FourierTests(unittest.TestCase):
    def test_known_mode_and_mean(self):
        for count in [63, 64]:
            with self.subTest(count=count):
                field = 7.0 + 3.0 * np.cos(2 * np.pi * 4 * np.arange(count) / count)
                fluctuation = pressure_fluctuation(field)
                power = one_sided_power(fluctuation)
                self.assertAlmostEqual(power[4], 4.5)
                self.assertAlmostEqual(power.sum(), np.var(field))
                self.assertAlmostEqual(float(fluctuation.mean()), 0.0)

    def test_nyquist_is_not_doubled(self):
        power = one_sided_power(2.0 * (-1.0) ** np.arange(64))
        self.assertAlmostEqual(power[-1], 4.0)
        self.assertAlmostEqual(power.sum(), 4.0)

    def test_random_parseval(self):
        rng = np.random.default_rng(8)
        for count in [31, 32]:
            field = pressure_fluctuation(rng.normal(size=(4, 6, count)))
            np.testing.assert_allclose(
                one_sided_power(field).sum(axis=-1), np.mean(field**2, axis=-1)
            )

    def test_variance_before_square_root(self):
        rms = weighted_rms(
            np.array([[1.0, 9.0]]), np.array([[1.0, 3.0]]), np.ones((1, 2), dtype=bool)
        )
        self.assertAlmostEqual(rms[0], np.sqrt(7.0))

    def test_flux_decomposition_and_sign(self):
        phase = 2 * np.pi * np.arange(64) / 64
        pressure = (10.0 + 2.0 * np.cos(phase))[None, None, :]
        velocity = (3.0 + 4.0 * np.cos(phase))[None, None, :]
        total, fluctuation = pressure_fluxes(pressure, velocity)
        self.assertAlmostEqual(total.item(), 34.0)
        self.assertAlmostEqual(fluctuation.item(), 4.0)
        np.testing.assert_allclose(
            total, pressure.mean(axis=-1) * velocity.mean(axis=-1) + fluctuation
        )
        _, negative = pressure_fluxes(pressure, -velocity)
        self.assertLess(negative.item(), 0.0)

    def test_layouts(self):
        padded = np.ones((4, 16))
        padded[:, 14:] = 0
        indices, _ = physical_y_indices(16, 12, 1, 6, padded)
        np.testing.assert_array_equal(indices, np.arange(1, 13))
        indices, _ = physical_y_indices(16, 12, 1, 6, np.ones((4, 16)))
        np.testing.assert_array_equal(indices, np.r_[1:7, 9:15])
        with self.assertRaises(ValueError):
            physical_y_indices(17, 12, 1, 6, np.ones((4, 17)))


class BPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="filament-test-")
        self.root = Path(self.temp.name)
        self.amplitude = write_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_domain_and_profiles(self):
        with DiagnosticRun(self.root) as run:
            self.assertTrue(run.geometry_verified)
            self.assertEqual(run.valid.sum(), 6 * 12)
            p = run.field("P", 1)
            fluctuation = pressure_fluctuation(p)
            np.testing.assert_allclose(np.mean(p, axis=-1), 0.02)
            np.testing.assert_allclose(
                np.mean(fluctuation**2, axis=-1), self.amplitude**2 / 2.0
            )
            profiles = radial_profiles(
                np.mean(fluctuation**2, axis=-1), run, run.profile_geometry()
            )
            self.assertTrue(np.all(np.isfinite(profiles["rms_closed_kpa"][1:5])))
            self.assertTrue(np.all(np.isnan(profiles["rms_closed_kpa"][5:])))
            self.assertTrue(np.all(np.isfinite(profiles["rms_sol_kpa"][5:7])))
            self.assertTrue(np.all(np.isfinite(profiles["rms_omp_kpa"][1:7])))

    def test_streamed_products_and_units(self):
        result = analyze(
            self.root, self.root / "output", stop=3, plots=False, require_profiles=True
        )
        with np.load(result) as data:
            self.assertEqual(data["modal_fraction"].shape, (3, 33))
            np.testing.assert_allclose(data["modal_fraction"][1:, 3], 1.0)
            self.assertEqual(data["n"][3], 15)
            self.assertTrue(np.all(data["modal_fraction"][0] == 0))
            expected = np.sum(
                self.amplitude**2
                / 2
                * (pressure_scale(2) / 1000) ** 2
                * data["weights_m3_per_rad"]
            )
            self.assertAlmostEqual(
                data["modal_power_kpa2_m3_per_rad"][1].sum(), expected
            )
            self.assertAlmostEqual(
                data["modal_power_kpa2_m3_per_rad"][2].sum(), 4 * expected
            )
            np.testing.assert_allclose(data["time_s"], [0, 1e-6, 2e-6])
        metadata = json.loads((result.parent / "metadata.json").read_text())
        self.assertEqual(metadata["profile_status"], "available")
        self.assertLess(metadata["max_parseval_relative_error"], 1e-12)
        with (result.parent / "summary.csv").open(newline="") as stream:
            summary = list(csv.DictReader(stream))
        self.assertAlmostEqual(
            float(summary[1]["mean_total_pressure_kpa"]),
            0.05 * pressure_scale(2.0) / 1000,
        )
        self.assertAlmostEqual(
            float(summary[1]["axisymmetric_perturbation_rms_kpa"]),
            0.02 * pressure_scale(2.0) / 1000,
        )

    def test_mismatched_grid_still_allows_spectra(self):
        with adios2.Stream(str(self.root / "wrong_grid.bp"), "w") as stream:
            stream.begin_step()
            stream.write("Rxy", np.ones((8, 12)))
            stream.end_step()
        result = analyze(
            self.root,
            self.root / "spectra_only",
            grid_path=self.root / "wrong_grid.bp",
            stop=3,
            plots=False,
        )
        with np.load(result) as data:
            self.assertNotIn("rms_omp_kpa", data)
        with self.assertRaisesRegex(ValueError, "pairing"):
            analyze(
                self.root,
                self.root / "strict",
                grid_path=self.root / "wrong_grid.bp",
                stop=3,
                require_profiles=True,
            )

    def test_numeric_grid_mismatch(self):
        with adios2.FileReader(str(self.root / "grid.bp")) as reader:
            grid = {name: reader.read(name) for name in reader.available_variables()}
        grid["Bxy"] *= 1.1
        wrong = self.root / "wrong_values.bp"
        with adios2.Stream(str(wrong), "w") as stream:
            stream.begin_step()
            for name, values in grid.items():
                stream.write(name, values)
            stream.end_step()
        with DiagnosticRun(self.root, wrong) as run:
            self.assertFalse(run.geometry_verified)
            with self.assertRaisesRegex(ValueError, "Bxy pairing"):
                run.require_geometry()
        with DiagnosticRun(self.root, wrong, grid_confirmed=True) as run:
            run.require_geometry()
            self.assertTrue(run.geometry_accepted)
            self.assertFalse(run.geometry_verified)
            self.assertEqual(
                run.audit["geometry_acceptance_basis"],
                "user-confirmed input-grid provenance",
            )

    def test_confirmation_still_requires_matching_topology(self):
        with adios2.FileReader(str(self.root / "grid.bp")) as reader:
            grid = {name: reader.read(name) for name in reader.available_variables()}
        grid["ixseps1"] = np.asarray(6)
        wrong = self.root / "wrong_topology.bp"
        with adios2.Stream(str(wrong), "w") as stream:
            stream.begin_step()
            for name, values in grid.items():
                stream.write(name, values)
            stream.end_step()
        with DiagnosticRun(self.root, wrong, grid_confirmed=True) as run:
            self.assertFalse(run.geometry_accepted)
            with self.assertRaisesRegex(ValueError, "topology ixseps1"):
                run.require_geometry()

    def test_overwrite_does_not_leave_old_plots(self):
        output = self.root / "old_output"
        output.mkdir()
        (output / "spectra.png").touch()
        with self.assertRaisesRegex(FileExistsError, "stale plots"):
            analyze(self.root, output, stop=3, plots=False, overwrite=True)

    def test_double_null_and_closing_sample(self):
        other = self.root / "double-null"
        other.mkdir()
        write_fixture(other, layout="double-null", closing_plane=True)
        with DiagnosticRun(other) as run:
            self.assertTrue(run.geometry_verified)
            self.assertEqual(run.field("P", 1).shape, (8, 12, 64))
            np.testing.assert_array_equal(run.y_indices, np.r_[1:7, 9:15])

    def test_zero_signal_and_default_cutoff(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            analyze(self.root, self.root / "empty", plots=False)
        result = analyze(self.root, self.root / "zero", start=0, stop=1, plots=False)
        with np.load(result) as data:
            self.assertEqual(data["modal_power_kpa2_m3_per_rad"].sum(), 0.0)
            self.assertEqual(data["rms_omp_kpa"][0, 3], 0.0)

    def test_renderer_quantities_and_signed_surfaces(self):
        from filament_renderer import _RunData, _build_filament_mesh, _selected_pressure

        data = _RunData.load(self.root)
        try:
            p, color, _, _ = data.read_step(1, "p_fluct_kpa")
            expected = pressure_fluctuation(p) * pressure_scale(2.0) / 1000
            np.testing.assert_allclose(color, expected)
            selected = _selected_pressure(p, "fluctuation", None, 1.0)
            self.assertEqual(
                _build_filament_mesh(data, selected, 100.0, allow_empty=True).n_cells, 0
            )
            self.assertGreater(_build_filament_mesh(data, selected, 1e-4).n_cells, 0)
            self.assertGreater(_build_filament_mesh(data, selected, -1e-4).n_cells, 0)
            p0 = np.full(p.shape[:2], 0.03)
            np.testing.assert_allclose(
                _selected_pressure(p, "total", p0, 1.0), p + 0.03
            )
        finally:
            data.close()


if __name__ == "__main__":
    unittest.main()
