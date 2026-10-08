"""Batch rendering for ELMO/BOUT++ filament visualizations.

The mesh construction follows VisIt's BOUT++ grid reader for a two-X-point
topology.  Each image contains one opaque, half-torus background mesh and one
opaque, full-torus pressure-isosurface mesh.

The public entry point is :func:`render_filaments`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import adios2
import numpy as np
import pyvista as pv


PRESSURE_HEATMAP = "RdBu_r"
FILAMENT_COLOR = "#9f1d20"
VISIT_NZ_OUT = 180
VISIT_FULL_K = np.arange(VISIT_NZ_OUT + 1, dtype=int)
VISIT_HALF_K = np.arange(VISIT_NZ_OUT // 2 + 1, dtype=int)
VISIT_HALF_PHI = VISIT_HALF_K * (2.0 * np.pi / VISIT_NZ_OUT)
VISIT_REPLICATION_ANGLE = np.pi / 24.0
CAMERA_PARALLEL_SCALE_FACTOR = 0.82


@dataclass
class _RunData:
    run_dir: Path
    bp_path: Path
    grid_path: Path
    bp_reader: object
    available: dict
    nsteps: int
    nx: int
    ny: int
    nz: int
    raw_ny: int
    myg: int
    physical_y: slice
    diffusion: float
    zperiod: int
    ixseps1: int
    ixseps2: int
    jyseps1_1: int
    jyseps1_2: int
    jyseps2_1: int
    jyseps2_2: int
    ny_inner: int
    rxy: np.ndarray
    zxy: np.ndarray
    zshift: np.ndarray
    shift_angle: np.ndarray
    dy: np.ndarray
    g22: np.ndarray
    prev_y: np.ndarray
    next_y: np.ndarray
    prev_shift: np.ndarray
    next_shift: np.ndarray
    y_start: np.ndarray
    x_index: np.ndarray
    kz: np.ndarray
    bout_blocks: list
    visit_mappings: list

    def close(self) -> None:
        if self.bp_reader is not None:
            self.bp_reader.close()
            self.bp_reader = None

    @classmethod
    def load(cls, run_dir: Path) -> "_RunData":
        run_dir = Path(run_dir).expanduser().resolve()
        bp_path = run_dir / "BOUT.dmp.bp"
        grid_path = run_dir / "grid.bp"
        if not bp_path.exists():
            raise FileNotFoundError(f"ADIOS output not found: {bp_path}")
        if not grid_path.exists():
            raise FileNotFoundError(f"ADIOS BOUT++ grid not found: {grid_path}")

        required = [
            "P",
            "dy",
            "g_22",
            "diffusion_par",
            "ny",
            "MYG",
            "zperiod",
            "ixseps1",
            "ixseps2",
            "jyseps1_1",
            "jyseps1_2",
            "jyseps2_1",
            "jyseps2_2",
            "ny_inner",
        ]
        bp = adios2.FileReader(str(bp_path))
        available = bp.available_variables()
        missing = [name for name in required if name not in available]
        if missing:
            bp.close()
            raise KeyError(f"Missing ADIOS variables in {bp_path}: {missing}")

        nsteps = int(available["P"]["AvailableStepsCount"])
        p_shape = tuple(
            int(value) for value in re.findall(r"\d+", available["P"]["Shape"])
        )
        if len(p_shape) != 3:
            raise ValueError(f"Expected a 3-D P shape, got {p_shape}")

        dy_raw = np.asarray(bp.read("dy"), dtype=np.float64).squeeze()
        g22_raw = np.asarray(bp.read("g_22"), dtype=np.float64).squeeze()
        diffusion = float(np.asarray(bp.read("diffusion_par")).squeeze())
        ny_model = int(np.asarray(bp.read("ny")).squeeze())
        myg = int(np.asarray(bp.read("MYG")).squeeze())
        zperiod = int(np.asarray(bp.read("zperiod")).squeeze())
        ixseps1 = int(np.asarray(bp.read("ixseps1")).squeeze())
        ixseps2 = int(np.asarray(bp.read("ixseps2")).squeeze())
        jyseps1_1 = int(np.asarray(bp.read("jyseps1_1")).squeeze())
        jyseps1_2 = int(np.asarray(bp.read("jyseps1_2")).squeeze())
        jyseps2_1 = int(np.asarray(bp.read("jyseps2_1")).squeeze())
        jyseps2_2 = int(np.asarray(bp.read("jyseps2_2")).squeeze())
        ny_inner = int(np.asarray(bp.read("ny_inner")).squeeze())

        with adios2.FileReader(str(grid_path)) as grid:
            grid_variables = grid.available_variables()
            grid_required = ["Rxy", "Zxy", "zShift", "ShiftAngle"]
            missing = [name for name in grid_required if name not in grid_variables]
            if missing:
                raise KeyError(f"Missing grid variables in {grid_path}: {missing}")
            rxy = np.asarray(grid.read("Rxy"), dtype=np.float64).copy()
            zxy = np.asarray(grid.read("Zxy"), dtype=np.float64).copy()
            zshift = np.asarray(grid.read("zShift"), dtype=np.float64).copy()
            shift_angle = np.asarray(
                grid.read("ShiftAngle"), dtype=np.float64
            ).squeeze()

        nx, ny = zshift.shape
        raw_nx, raw_ny, nz = p_shape
        physical_y = slice(myg, myg + ny)
        dy = dy_raw[:, physical_y]
        g22 = g22_raw[:, physical_y]

        if raw_nx != nx:
            raise ValueError(f"P nx={raw_nx} does not match grid nx={nx}")
        if ny != ny_model:
            raise ValueError(f"Grid ny={ny} does not match ADIOS ny={ny_model}")
        if raw_ny < myg + ny:
            raise ValueError(
                f"P raw y extent {raw_ny} cannot provide physical slice "
                f"{myg}:{myg + ny}"
            )
        if not (
            rxy.shape == zxy.shape == zshift.shape == dy.shape == g22.shape
        ):
            raise ValueError("Grid geometry and metric arrays have inconsistent shapes")
        if shift_angle.shape != (nx,):
            raise ValueError(
                f"Expected ShiftAngle shape {(nx,)}, got {shift_angle.shape}"
            )
        if np.any(g22 <= 0.0):
            raise ValueError("g_22 must be positive")
        if ixseps1 != ixseps2:
            raise ValueError(
                "This renderer currently requires ixseps1 == ixseps2; "
                f"got {ixseps1} and {ixseps2}"
            )

        prev_y, next_y, prev_shift, next_shift = _make_parallel_topology(
            nx,
            ny,
            ny_inner,
            ixseps1,
            jyseps1_1,
            jyseps1_2,
            jyseps2_1,
            jyseps2_2,
            shift_angle,
        )
        z_length = 2.0 * np.pi / zperiod
        kz = 2.0 * np.pi * np.arange(nz // 2 + 1) / z_length
        y_start = np.broadcast_to(np.arange(ny, dtype=np.int64), (nx, ny))
        x_index = np.arange(nx, dtype=np.int64)[:, None]

        y_mid = jyseps2_1 + (jyseps1_2 - jyseps2_1) // 2 + 1
        inner_x = np.arange(0, ixseps1 + 1, dtype=int)
        outer_x = np.arange(ixseps1, nx, dtype=int)
        bout_blocks = [
            (
                "private_flux",
                inner_x,
                np.r_[0 : jyseps1_1 + 1, jyseps2_2 + 1 : ny],
            ),
            ("lower_sol", outer_x, np.arange(0, y_mid, dtype=int)),
            ("upper_sol", outer_x, np.arange(y_mid, ny, dtype=int)),
            (
                "lower_core",
                inner_x,
                np.arange(jyseps1_1 + 1, y_mid, dtype=int),
            ),
            (
                "upper_core",
                inner_x,
                np.r_[y_mid : jyseps2_2 + 1, jyseps1_1 + 1],
            ),
        ]

        data = cls(
            run_dir=run_dir,
            bp_path=bp_path,
            grid_path=grid_path,
            bp_reader=bp,
            available=available,
            nsteps=nsteps,
            nx=nx,
            ny=ny,
            nz=nz,
            raw_ny=raw_ny,
            myg=myg,
            physical_y=physical_y,
            diffusion=diffusion,
            zperiod=zperiod,
            ixseps1=ixseps1,
            ixseps2=ixseps2,
            jyseps1_1=jyseps1_1,
            jyseps1_2=jyseps1_2,
            jyseps2_1=jyseps2_1,
            jyseps2_2=jyseps2_2,
            ny_inner=ny_inner,
            rxy=rxy,
            zxy=zxy,
            zshift=zshift,
            shift_angle=shift_angle,
            dy=dy,
            g22=g22,
            prev_y=prev_y,
            next_y=next_y,
            prev_shift=prev_shift,
            next_shift=next_shift,
            y_start=y_start,
            x_index=x_index,
            kz=kz,
            bout_blocks=bout_blocks,
            visit_mappings=[],
        )
        data.visit_mappings = _make_visit_mappings(data)
        return data

    def _shift_in_z(self, values: np.ndarray, delta_z: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(values, axis=-1)
        phase = np.exp(1j * delta_z[..., None] * self.kz)
        return np.fft.irfft(spectrum * phase, n=self.nz, axis=-1)

    def _shifted_neighbor(
        self,
        values: np.ndarray,
        mapping: np.ndarray,
        branch_shift: np.ndarray,
    ) -> np.ndarray:
        proposed_y = mapping[self.x_index, self.y_start]
        at_boundary = proposed_y < 0
        current_y = np.where(at_boundary, self.y_start, proposed_y)
        accumulated_shift = np.where(
            at_boundary, 0.0, branch_shift[self.x_index, self.y_start]
        )
        neighbor_values = values[self.x_index, current_y, :]
        effective_zshift = self.zshift[self.x_index, current_y] + accumulated_shift
        delta_z = effective_zshift - self.zshift
        return self._shift_in_z(neighbor_values, delta_z)

    def read_step(
        self, step: int, color_field: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
        if not 0 <= step < self.nsteps:
            raise IndexError(
                f"step={step} is outside the available range 0:{self.nsteps - 1}"
            )
        if color_field not in self.available:
            raise KeyError(f"ADIOS variable not found: {color_field}")

        bp = self.bp_reader
        pressure = _read_adios_variable(bp, self.available, "P", step)
        color = _read_adios_variable(
            bp, self.available, color_field, step, allow_static=True
        )
        if "t_array" in self.available:
            time_values = _read_adios_variable(
                bp, self.available, "t_array", step, allow_static=True
            )
            simulation_time = float(np.asarray(time_values).reshape(-1)[0])
        else:
            simulation_time = None

        pressure = self._physical_field(pressure, "P")
        color = self._physical_field(color, color_field)
        if pressure.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Expected P shape {(self.nx, self.ny, self.nz)}, "
                f"got {pressure.shape}"
            )

        pressure_m1 = self._shifted_neighbor(
            pressure, self.prev_y, self.prev_shift
        )
        pressure_p1 = self._shifted_neighbor(
            pressure, self.next_y, self.next_shift
        )
        # Centered difference: second-order accurate at interior y points.
        derivative_y = (pressure_p1 - pressure_m1) / (
            2.0 * self.dy[..., None]
        )
        grad_parallel = derivative_y / np.sqrt(self.g22)[..., None]
        heatflux_parallel_e = -self.diffusion * grad_parallel
        if not np.all(np.isfinite(heatflux_parallel_e)):
            raise FloatingPointError(
                f"Non-finite heatflux_par_e values at step {step}"
            )
        return pressure, color, heatflux_parallel_e, simulation_time

    def _physical_field(self, values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).squeeze()
        if values.ndim not in (2, 3):
            raise ValueError(f"Unexpected {name} shape: {values.shape}")
        if values.shape[0] != self.nx:
            raise ValueError(
                f"{name} nx={values.shape[0]} does not match grid nx={self.nx}"
            )
        if values.shape[1] == self.raw_ny:
            values = values[:, self.physical_y, ...]
        elif values.shape[1] != self.ny:
            raise ValueError(
                f"{name} y extent {values.shape[1]} is neither raw "
                f"{self.raw_ny} nor physical {self.ny}"
            )
        return values


def _read_adios_variable(
    bp,
    available: dict,
    name: str,
    step: int,
    *,
    allow_static: bool = False,
) -> np.ndarray:
    step_count = int(available[name].get("AvailableStepsCount", "1"))
    if allow_static and step_count <= 1:
        values = bp.read(name)
    else:
        values = bp.read(name, step_selection=[step, 1])
    values = np.asarray(values)
    if values.ndim >= 1 and values.shape[0] == 1:
        shape_text = available[name].get("Shape", "")
        declared_shape = tuple(int(value) for value in re.findall(r"\d+", shape_text))
        if len(declared_shape) + 1 == values.ndim:
            values = values[0]
    return values


def get_simulation_time(run_dir: str | Path, step: int) -> float:
    """Return the simulation time stored in an ELMO timestep."""
    run_dir = Path(run_dir).expanduser().resolve()
    bp_path = run_dir / "BOUT.dmp.bp"
    if not bp_path.exists():
        raise FileNotFoundError(f"ADIOS output not found: {bp_path}")

    with adios2.FileReader(str(bp_path)) as bp:
        available = bp.available_variables()
        if "t_array" not in available:
            raise KeyError(f"Time variable 't_array' not found in {bp_path}")
        step_count = int(available["t_array"].get("AvailableStepsCount", "1"))
        if not 0 <= step < step_count:
            raise IndexError(
                f"step={step} is outside the available range 0:{step_count - 1}"
            )
        values = _read_adios_variable(
            bp, available, "t_array", step, allow_static=True
        )

    values = np.asarray(values).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar time at step {step}, got {values.shape}")
    return float(values[0])


def _make_parallel_topology(
    nx: int,
    ny: int,
    ny_inner: int,
    ixseps: int,
    jyseps1_1: int,
    jyseps1_2: int,
    jyseps2_1: int,
    jyseps2_2: int,
    shift_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev_y = np.broadcast_to(np.arange(ny, dtype=np.int64) - 1, (nx, ny)).copy()
    next_y = np.broadcast_to(np.arange(ny, dtype=np.int64) + 1, (nx, ny)).copy()
    prev_y[:, 0] = -1
    next_y[:, ny_inner - 1] = -1
    prev_y[:, ny_inner] = -1
    next_y[:, ny - 1] = -1

    prev_shift = np.zeros((nx, ny), dtype=np.float64)
    next_shift = np.zeros((nx, ny), dtype=np.float64)
    inner_x = slice(0, ixseps)

    prev_y[inner_x, jyseps1_1 + 1] = jyseps2_2
    next_y[inner_x, jyseps2_2] = jyseps1_1 + 1
    prev_shift[inner_x, jyseps1_1 + 1] = -shift_angle[inner_x]
    next_shift[inner_x, jyseps2_2] = shift_angle[inner_x]

    next_y[inner_x, jyseps1_1] = jyseps2_2 + 1
    prev_y[inner_x, jyseps2_2 + 1] = jyseps1_1

    next_y[inner_x, jyseps2_1] = jyseps1_2 + 1
    prev_y[inner_x, jyseps1_2 + 1] = jyseps2_1

    prev_y[inner_x, jyseps2_1 + 1] = jyseps1_2
    next_y[inner_x, jyseps1_2] = jyseps2_1 + 1
    return prev_y, next_y, prev_shift, next_shift


def _visit_subgrid_specs(data: _RunData) -> list[dict]:
    mid_y = data.jyseps2_1 + (data.jyseps1_2 - data.jyseps2_1) // 2 + 1
    return [
        dict(
            name="private_flux",
            istart=0,
            iend=data.ixseps1 + 1,
            branches=[(0, data.jyseps1_1 + 1), (data.jyseps2_2 + 1, data.ny)],
        ),
        dict(
            name="lower_sol",
            istart=data.ixseps1,
            iend=data.nx,
            branches=[(0, mid_y)],
        ),
        dict(
            name="upper_sol",
            istart=data.ixseps1,
            iend=data.nx,
            branches=[(mid_y, data.ny)],
        ),
        dict(
            name="lower_core",
            istart=0,
            iend=data.ixseps1 + 1,
            branches=[
                (data.jyseps1_1 + 1, data.jyseps2_1 + 1),
                (data.jyseps1_2 + 1, data.jyseps1_2 + 2),
            ],
        ),
        dict(
            name="upper_core",
            istart=0,
            iend=data.ixseps1 + 1,
            branches=[
                (data.jyseps1_2 + 1, data.jyseps2_2 + 1),
                (data.jyseps1_1 + 1, data.jyseps1_1 + 2),
            ],
        ),
        dict(
            name="lower_xpoint",
            istart=data.ixseps1,
            iend=data.ixseps1 + 1,
            branches=[
                (data.jyseps1_1, data.jyseps1_1 + 2),
                (data.jyseps2_2 + 1, data.jyseps2_2 - 1),
            ],
        ),
        dict(
            name="upper_xpoint",
            istart=data.ixseps1,
            iend=data.ixseps1 + 1,
            branches=[
                (data.jyseps2_1, data.jyseps2_1 + 2),
                (data.jyseps1_2 + 1, data.jyseps1_2 - 1),
            ],
        ),
    ]


def _prepare_visit_subgrid(data: _RunData, spec: dict) -> dict:
    grid = dict(spec)
    grid["jindex"] = np.concatenate(
        [
            np.arange(start, end, 1 if start < end else -1, dtype=int)
            for start, end in grid["branches"]
        ]
    )
    grid["nxIn"] = grid["iend"] - grid["istart"]
    grid["nyIn"] = sum(abs(end - start) for start, end in grid["branches"])
    grid["special"] = (
        len(grid["branches"]) == 2 and grid["istart"] + 1 == grid["iend"]
    )
    if grid["special"]:
        grid["nxIn"] = 2
        grid["nyIn"] = 2

    ijindex = []
    for i in range(grid["istart"], grid["iend"]):
        for start, end in grid["branches"]:
            step = 1 if start < end else -1
            for j in range(start, end, step):
                ijindex.append(i * data.ny + j)
    grid["ijindex"] = np.asarray(ijindex, dtype=int).reshape(
        grid["nxIn"], grid["nyIn"]
    )

    grid["jnrep"] = np.empty(grid["nyIn"] - 1, dtype=int)
    for j in range(grid["nyIn"] - 1):
        j1 = grid["jindex"][j]
        j2 = grid["jindex"][j + 1]
        delta = np.max(
            np.abs(
                data.zshift[grid["istart"] : grid["iend"], j2]
                - data.zshift[grid["istart"] : grid["iend"], j1]
            )
        )
        grid["jnrep"][j] = int(
            np.clip(np.ceil(delta / VISIT_REPLICATION_ANGLE), 6, 12)
        )

    grid["inrep"] = np.ones(grid["nxIn"] - 1, dtype=int)
    if not grid["special"]:
        for local_i, i in enumerate(range(grid["istart"], grid["iend"] - 1)):
            jj = grid["jindex"]
            delta = np.max(np.abs(data.zshift[i + 1, jj] - data.zshift[i, jj]))
            grid["inrep"][local_i] = int(
                np.clip(np.ceil(delta / VISIT_REPLICATION_ANGLE), 1, 12)
            )

    if grid["special"]:
        grid["nxOut"] = 2
        grid["nyOut"] = 2
    else:
        grid["nxOut"] = int(grid["inrep"].sum()) + 1
        grid["nyOut"] = int(grid["jnrep"].sum()) + 1
    return grid


def _match_xpoint_replication(data: _RunData, subgrids: list[dict]) -> None:
    nx_max = max(
        subgrids[1]["jnrep"][data.jyseps1_1],
        subgrids[2]["jnrep"][data.jyseps2_2 - data.jyseps2_1 - 6],
    )
    ny_max = max(
        subgrids[0]["jnrep"][data.jyseps1_1],
        subgrids[4]["jnrep"][subgrids[4]["nyIn"] - 2],
    )
    index = data.jyseps1_1
    subgrids[1]["nyOut"] += int(nx_max - subgrids[1]["jnrep"][index])
    subgrids[1]["jnrep"][index] = nx_max
    index = data.jyseps2_2 - data.jyseps2_1 - 5
    subgrids[2]["nyOut"] += int(nx_max - subgrids[2]["jnrep"][index])
    subgrids[2]["jnrep"][index] = nx_max
    index = data.jyseps1_1
    subgrids[0]["nyOut"] += int(ny_max - subgrids[0]["jnrep"][index])
    subgrids[0]["jnrep"][index] = ny_max
    index = subgrids[4]["nyIn"] - 2
    subgrids[4]["nyOut"] += int(ny_max - subgrids[4]["jnrep"][index])
    subgrids[4]["jnrep"][index] = ny_max
    subgrids[5]["nxOut"] = int(nx_max) + 1
    subgrids[5]["inrep"][0] = nx_max
    subgrids[5]["nyOut"] = int(ny_max) + 1
    subgrids[5]["jnrep"][0] = ny_max

    nx_max = max(
        subgrids[1]["jnrep"][data.jyseps2_1],
        subgrids[2]["jnrep"][(data.jyseps1_2 - data.jyseps2_1) // 2 - 1],
    )
    ny_max = subgrids[3]["jnrep"][subgrids[3]["nyIn"] - 2]
    index = data.jyseps2_1
    subgrids[1]["nyOut"] += int(nx_max - subgrids[1]["jnrep"][index])
    subgrids[1]["jnrep"][index] = nx_max
    index = (data.jyseps1_2 - data.jyseps2_1) // 2 - 1
    subgrids[2]["nyOut"] += int(nx_max - subgrids[2]["jnrep"][index])
    subgrids[2]["jnrep"][index] = nx_max
    subgrids[6]["nxOut"] = int(nx_max) + 1
    subgrids[6]["inrep"][0] = nx_max
    subgrids[6]["nyOut"] = int(ny_max) + 1
    subgrids[6]["jnrep"][0] = ny_max


def _build_visit_mapping(data: _RunData, grid: dict) -> dict:
    shape = (grid["nxOut"], grid["nyOut"])
    indices = np.empty((4,) + shape, dtype=np.int32)
    value_weights = np.empty((4,) + shape, dtype=np.float32)
    shift_weights = np.empty((4,) + shape, dtype=np.float32)

    if grid["special"]:
        corners = grid["ijindex"].ravel()
        for ii in range(grid["nxOut"]):
            fraction_i = ii / (grid["nxOut"] - 1)
            for jj in range(grid["nyOut"]):
                fraction_j = jj / (grid["nyOut"] - 1)
                indices[:, ii, jj] = corners
                value_weights[:, ii, jj] = [
                    (1.0 - fraction_i) * (1.0 - fraction_j),
                    fraction_i * (1.0 - fraction_j),
                    (1.0 - fraction_i) * fraction_j,
                    fraction_i * fraction_j,
                ]
                # VisIt CreateVar interpolates zShift in this parametric order.
                shift_weights[:, ii, jj] = [
                    (1.0 - fraction_i) * (1.0 - fraction_j),
                    (1.0 - fraction_i) * fraction_j,
                    fraction_i * (1.0 - fraction_j),
                    fraction_i * fraction_j,
                ]
    else:
        sum_i = 0
        for i in range(grid["nxIn"] - 1):
            sum_j = 0
            for j in range(grid["nyIn"] - 1):
                corners = np.asarray(
                    [
                        grid["ijindex"][i, j],
                        grid["ijindex"][i, j + 1],
                        grid["ijindex"][i + 1, j],
                        grid["ijindex"][i + 1, j + 1],
                    ],
                    dtype=np.int32,
                )
                for ii in range(grid["inrep"][i] + 1):
                    fraction_i = ii / grid["inrep"][i]
                    for jj in range(grid["jnrep"][j] + 1):
                        fraction_j = jj / grid["jnrep"][j]
                        output_i = sum_i + ii
                        output_j = sum_j + jj
                        indices[:, output_i, output_j] = corners
                        weights = [
                            (1.0 - fraction_i) * (1.0 - fraction_j),
                            (1.0 - fraction_i) * fraction_j,
                            fraction_i * (1.0 - fraction_j),
                            fraction_i * fraction_j,
                        ]
                        value_weights[:, output_i, output_j] = weights
                        shift_weights[:, output_i, output_j] = weights
                sum_j += grid["jnrep"][j]
            sum_i += grid["inrep"][i]

    flat_r = data.rxy.reshape(-1)
    flat_z = data.zxy.reshape(-1)
    flat_shift = data.zshift.reshape(-1)
    radius = np.sum(value_weights * flat_r[indices], axis=0)
    vertical = np.sum(value_weights * flat_z[indices], axis=0)
    shift = np.sum(shift_weights * flat_shift[indices], axis=0)
    return dict(
        name=grid["name"],
        indices=indices,
        value_weights=value_weights,
        shift=shift,
        radius=radius,
        vertical=vertical,
        shape=shape,
    )


def _make_visit_mappings(data: _RunData) -> list[dict]:
    subgrids = [
        _prepare_visit_subgrid(data, spec) for spec in _visit_subgrid_specs(data)
    ]
    _match_xpoint_replication(data, subgrids)
    return [_build_visit_mapping(data, grid) for grid in subgrids]


def _visit_interpolate_field(
    data: _RunData,
    mapping: dict,
    field_values: np.ndarray,
    plane_indices: np.ndarray = VISIT_HALF_K,
) -> np.ndarray:
    values = np.asarray(field_values)
    indices = mapping["indices"]
    weights = mapping["value_weights"]

    if values.ndim == 2:
        flat = values.reshape(-1)
        spatial = np.sum(weights * flat[indices], axis=0).astype(np.float32)
        return np.broadcast_to(
            spatial[..., None], spatial.shape + (len(plane_indices),)
        )
    if values.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D nodal field, got {values.shape}")

    source_nz = values.shape[2]
    flat = values.reshape(data.nx * data.ny, source_nz)
    result = np.empty(mapping["shape"] + (len(plane_indices),), dtype=np.float32)
    z_period = 2.0 * np.pi / data.zperiod
    source_spacing = z_period / source_nz

    for plane, k in enumerate(plane_indices):
        z_angle = k * 2.0 * np.pi / VISIT_NZ_OUT
        angle = np.mod(z_angle - mapping["shift"], z_period)
        source_coordinate = angle / source_spacing
        lower_unwrapped = np.floor(source_coordinate).astype(np.int32)
        lower = lower_unwrapped % source_nz
        upper = np.ceil(source_coordinate).astype(np.int32) % source_nz
        fraction = source_coordinate - lower_unwrapped
        first = np.sum(weights * flat[indices, lower[None, ...]], axis=0)
        second = np.sum(weights * flat[indices, upper[None, ...]], axis=0)
        result[..., plane] = first + (second - first) * fraction
    return result


def _structured_face(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    scalars: np.ndarray,
    field_name: str,
) -> pv.StructuredGrid:
    x = np.asarray(x, dtype=np.float32)
    y = np.broadcast_to(np.asarray(y, dtype=np.float32), x.shape)
    z = np.broadcast_to(np.asarray(z, dtype=np.float32), x.shape)
    grid = pv.StructuredGrid(x[..., None], y[..., None], z[..., None])
    grid.point_data[field_name] = np.asarray(scalars, dtype=np.float32).ravel(
        order="F"
    )
    return grid


def _visit_domain_boundary(
    data: _RunData, mapping: dict, field_values: np.ndarray, field_name: str
) -> list[pv.StructuredGrid]:
    radius = mapping["radius"]
    vertical = mapping["vertical"]
    values = _visit_interpolate_field(data, mapping, field_values)
    faces = []

    for plane in (0, len(VISIT_HALF_PHI) - 1):
        theta = VISIT_HALF_PHI[plane]
        faces.append(
            _structured_face(
                (radius * np.cos(theta)).T,
                vertical.T,
                (radius * np.sin(theta)).T,
                values[..., plane].T,
                field_name,
            )
        )

    cos_theta = np.cos(VISIT_HALF_PHI)[None, :]
    sin_theta = np.sin(VISIT_HALF_PHI)[None, :]
    for j in (0, radius.shape[1] - 1):
        r = radius[:, j, None]
        faces.append(
            _structured_face(
                r * cos_theta,
                vertical[:, j, None],
                r * sin_theta,
                values[:, j, :],
                field_name,
            )
        )
    for i in (0, radius.shape[0] - 1):
        r = radius[i, :, None]
        faces.append(
            _structured_face(
                r * cos_theta,
                vertical[i, :, None],
                r * sin_theta,
                values[i, :, :],
                field_name,
            )
        )
    return faces


def _merge_meshes(meshes: list[pv.DataSet]) -> pv.DataSet:
    if not meshes:
        raise ValueError("Cannot merge an empty mesh list")
    if hasattr(pv, "merge"):
        return pv.merge(meshes, merge_points=False)
    merged = meshes[0].copy()
    for mesh in meshes[1:]:
        merged = merged.merge(mesh, merge_points=False)
    return merged


def _make_sector_grid_geometry(
    data: _RunData, ix: np.ndarray, jy: np.ndarray
) -> pv.StructuredGrid:
    ix = np.asarray(ix, dtype=int)
    jy = np.asarray(jy, dtype=int)
    radius = data.rxy[np.ix_(ix, jy)]
    vertical = data.zxy[np.ix_(ix, jy)]
    shift = data.zshift[np.ix_(ix, jy)]

    phi = np.linspace(0.0, 2.0 * np.pi / data.zperiod, data.nz + 1)
    theta = shift[..., None] + phi[None, None, :]
    x = radius[..., None] * np.cos(theta)
    y = np.broadcast_to(vertical[..., None], theta.shape)
    z = radius[..., None] * np.sin(theta)
    return pv.StructuredGrid(x, y, z)


def _make_visit_grid(
    data: _RunData,
    mapping: dict,
    field_values: np.ndarray | None = None,
    field_name: str = "",
) -> pv.StructuredGrid:
    radius = mapping["radius"][..., None]
    vertical = mapping["vertical"][..., None]
    phi = VISIT_FULL_K * (2.0 * np.pi / VISIT_NZ_OUT)
    x = radius * np.cos(phi)[None, None, :]
    y = np.broadcast_to(vertical, x.shape)
    z = radius * np.sin(phi)[None, None, :]
    grid = pv.StructuredGrid(x, y, z)

    if field_values is not None:
        values = _visit_interpolate_field(
            data, mapping, field_values, VISIT_FULL_K
        )
        grid.point_data[field_name] = values.ravel(order="F")
    return grid


def _make_sector_grid(
    data: _RunData,
    pressure: np.ndarray,
    ix: np.ndarray,
    jy: np.ndarray,
) -> pv.StructuredGrid:
    grid = _make_sector_grid_geometry(data, ix, jy)

    pressure_values = pressure[np.ix_(ix, jy, np.arange(data.nz, dtype=int))]
    pressure_values = np.concatenate(
        (pressure_values, pressure_values[..., :1]), axis=2
    )
    grid.point_data["P"] = pressure_values.ravel(order="F")
    return grid


def _write_boutpp_grid(
    data: _RunData,
    output_path: Path,
    overwrite: bool,
    field_values: np.ndarray | None = None,
    field_name: str = "",
) -> Path:
    if output_path.exists() and not overwrite:
        print(f"  grid exists, skipping {output_path}", flush=True)
        return output_path

    blocks = pv.MultiBlock()
    for mapping in data.visit_mappings:
        blocks[mapping["name"]] = _make_visit_grid(
            data, mapping, field_values, field_name
        )
    blocks.save(output_path)
    print(
        f"  saved BOUT++ grid {output_path} "
        f"({len(blocks)} VisIt-compatible subgrids)",
        flush=True,
    )
    return output_path


def _build_filament_mesh(
    data: _RunData, pressure: np.ndarray, level: float
) -> pv.DataSet:
    sector_angle = 360.0 / data.zperiod
    surfaces = []
    for _, ix, jy in data.bout_blocks:
        shifted = _make_sector_grid(data, pressure, ix, jy)
        block_pressure = np.asarray(shifted.point_data["P"])
        if np.nanmin(block_pressure) <= level <= np.nanmax(block_pressure):
            filament = shifted.contour([level], scalars="P").triangulate()
            if filament.n_cells:
                for sector in range(data.zperiod):
                    if sector == 0:
                        rotated = filament.copy(deep=True)
                    else:
                        rotated = filament.rotate_y(
                            sector_angle * sector, inplace=False
                        )
                    if rotated.n_cells:
                        surfaces.append(rotated)
        del shifted
    if not surfaces:
        raise RuntimeError(f"No filament cells found at P={level:.12e}")
    return _merge_meshes(surfaces)


def _build_background_mesh(
    data: _RunData, field_values: np.ndarray, field_name: str
) -> pv.DataSet:
    faces = []
    for mapping in data.visit_mappings:
        faces.extend(_visit_domain_boundary(data, mapping, field_values, field_name))
    return _merge_meshes(faces)


def _background_scalars(
    data: _RunData, field_values: np.ndarray
) -> np.ndarray:
    values = []
    for mapping in data.visit_mappings:
        interpolated = _visit_interpolate_field(data, mapping, field_values)
        for plane in (0, len(VISIT_HALF_PHI) - 1):
            values.append(interpolated[..., plane].T.ravel(order="F"))
        for j in (0, interpolated.shape[1] - 1):
            values.append(interpolated[:, j, :])
            values[-1] = values[-1].ravel(order="F")
        for i in (0, interpolated.shape[0] - 1):
            values.append(interpolated[i, :, :].ravel(order="F"))
    return np.concatenate(values).astype(np.float32, copy=False)


def _update_background_scalars(
    data: _RunData,
    background: pv.DataSet,
    field_values: np.ndarray,
    field_name: str,
) -> None:
    scalars = _background_scalars(data, field_values)
    if len(scalars) != background.n_points:
        raise ValueError(
            f"Background scalar count {len(scalars)} does not match "
            f"mesh points {background.n_points}"
        )
    background.point_data[field_name] = scalars


def _automatic_color_limits(
    field_values: np.ndarray, field_name: str, percentile: float
) -> tuple[float, float]:
    finite = np.asarray(field_values)[np.isfinite(field_values)]
    if finite.size == 0:
        raise ValueError(f"{field_name} contains no finite values")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum < 0.0 < maximum:
        limit = float(np.percentile(np.abs(finite), percentile))
        if limit <= 0.0:
            limit = max(abs(minimum), abs(maximum))
        return -limit, limit
    if minimum == maximum:
        padding = abs(minimum) * 1.0e-6 or 1.0
        return minimum - padding, maximum + padding
    return minimum, maximum


def _render_frame(
    data: _RunData,
    background: pv.DataSet,
    pressure: np.ndarray,
    color_values: np.ndarray,
    color_field: str,
    iso_level: float,
    simulation_time: float | None,
    output_path: Path,
    color_limits: tuple[float, float],
    window_size: tuple[int, int],
    plotter: pv.Plotter,
    background_actor: object | None,
    filament_actor: object | None,
) -> tuple[int, int, object, object]:
    filaments = _build_filament_mesh(data, pressure, iso_level)

    if filament_actor is None:
        plotter.set_background("white")
        background_actor = plotter.add_mesh(
            background,
            scalars=color_field,
            preference="point",
            cmap=PRESSURE_HEATMAP,
            clim=color_limits,
            opacity=1.0,
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            interpolate_before_map=True,
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title=color_field,
                vertical=True,
                position_x=0.88,
                position_y=0.12,
                height=0.76,
                width=0.025,
                color="black",
                fmt="%.1e",
            ),
        )
        filament_actor = plotter.add_mesh(
            filaments,
            color=FILAMENT_COLOR,
            opacity=1.0,
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            show_scalar_bar=False,
        )
        mesh_radius = float(np.nanmax(np.abs(data.rxy)))
        vertical_span = float(np.nanmax(data.zxy) - np.nanmin(data.zxy))
        plotter.enable_parallel_projection()
        plotter.camera_position = [
            (5.0, 1.8, -9.060019493103027),
            (0.0, -0.017590701580047607, 0.0),
            (0.0, 1.0, 0.0),
        ]
        plotter.camera.parallel_scale = CAMERA_PARALLEL_SCALE_FACTOR * max(
            mesh_radius, 0.5 * vertical_span
        )
    else:
        background_actor.mapper.scalar_range = color_limits
        background_actor.mapper.Modified()
        filament_actor.mapper.SetInputData(filaments)
        filament_actor.mapper.Modified()
    if simulation_time is not None:
        if hasattr(plotter, "_time_text_actor"):
            plotter._time_text_actor.SetText(2, f"t = {simulation_time:08.2f}")
        else:
            plotter._time_text_actor = plotter.add_text(
                f"t = {simulation_time:08.2f}",
                position="upper_left",
                font_size=18,
                color="black",
            )
    plotter.show(screenshot=str(output_path), auto_close=False)
    counts = (int(background.n_cells), int(filaments.n_cells))
    del filaments
    gc.collect()
    return (*counts, background_actor, filament_actor)


def _normalize_inputs(
    input_dirs: str | Path | Sequence[str | Path] | Mapping[str, str | Path],
) -> list[tuple[str, Path]]:
    if isinstance(input_dirs, Mapping):
        items = [(str(name), Path(path)) for name, path in input_dirs.items()]
    elif isinstance(input_dirs, (str, Path)):
        path = Path(input_dirs)
        items = [(path.name, path)]
    else:
        paths = [Path(path) for path in input_dirs]
        items = [(f"run_{index:03d}_{path.name}", path) for index, path in enumerate(paths)]
    if not items:
        raise ValueError("input_dirs is empty")

    normalized = []
    seen = set()
    for name, path in items:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        if not safe_name:
            raise ValueError(f"Invalid run name: {name!r}")
        if safe_name in seen:
            raise ValueError(f"Duplicate output run name: {safe_name}")
        seen.add(safe_name)
        normalized.append((safe_name, path.expanduser()))
    return normalized


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "run",
        "input_dir",
        "filename",
        "step",
        "simulation_time",
        "iso_level",
        "iso_fraction",
        "P_min",
        "P_max",
        "heatflux_par_e_min",
        "heatflux_par_e_max",
        "color_field",
        "color_min",
        "color_max",
        "background_cells",
        "filament_cells",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_filaments(
    input_dirs: str | Path | Sequence[str | Path] | Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    step_start: int = 500,
    step_stop: int | None = None,
    step_stride: int = 10,
    iso_fraction: float | None = 0.50,
    iso_value: float | None = None,
    color_field: str = "phi",
    color_percentile: float = 98.0,
    color_limits: tuple[float, float] | None = None,
    filename_pattern: str = "filament.{step:05d}.png",
    write_grid_vtk: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = True,
    window_size: tuple[int, int] = (1200, 700),
    grid_output: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Render filament images for one or more ELMO ADIOS runs.

    Parameters
    ----------
    input_dirs
        One run directory, a sequence of run directories, or a mapping from
        short run names to directories.  A mapping is recommended for multiple
        runs.  Each directory must contain ``BOUT.dmp.bp`` and ``grid.bp``.
    output_dir
        Parent output directory.  Images are written below one subdirectory per
        run, for example ``output_dir/run5/filament.00500.png``.
    step_start, step_stop, step_stride
        Python-style timestep range.  ``step_stop=None`` uses all available
        steps, and the stop value is exclusive.
    iso_fraction
        Positive pressure-isosurface level as a fraction of ``max(P)`` at each
        timestep.  The default is 0.50.
    iso_value
        Absolute pressure-isosurface value.  Set this instead of
        ``iso_fraction`` to keep the level fixed across timesteps.
    color_field
        ADIOS variable used to color the opaque half-torus mesh.
    color_percentile
        Symmetric percentile used for signed fields such as ``phi``.
    color_limits
        Optional fixed ``(minimum, maximum)`` color range.
    filename_pattern
        ``str.format`` pattern receiving ``run``, ``step``, and ``time``.
    write_grid_vtk
        Write the seven VisIt-compatible BOUT++ structured subgrids to
        ``boutpp-grid.vtm`` below each run's output directory.
    overwrite
        Replace existing images when true.  Existing images are otherwise
        treated as completed frames.
    continue_on_error
        Record a failed frame in ``manifest.csv`` and continue when true.
    grid_output
        Optional VTK output path for the generated background visualization
        mesh.  The mesh is exported once per run after the first successful
        timestep; if omitted, it is written as ``background.vtk`` below the
        run output directory.

    Returns
    -------
    dict
        Mapping from run name to all successfully rendered or pre-existing
        image paths requested by this invocation.
    """
    if step_start < 0:
        raise ValueError("step_start must be non-negative")
    if step_stride <= 0:
        raise ValueError("step_stride must be positive")
    if (iso_fraction is None) == (iso_value is None):
        raise ValueError("Set exactly one of iso_fraction or iso_value")
    if iso_fraction is not None and not 0.0 < iso_fraction <= 1.0:
        raise ValueError("iso_fraction must be in the interval (0, 1]")
    if not 0.0 < color_percentile <= 100.0:
        raise ValueError("color_percentile must be in the interval (0, 100]")
    if color_limits is not None and color_limits[0] >= color_limits[1]:
        raise ValueError("color_limits must be increasing")
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("window_size must contain two positive integers")

    runs = _normalize_inputs(input_dirs)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}

    for run_name, run_dir in runs:
        print(f"Loading {run_name}: {run_dir}", flush=True)
        data = _RunData.load(run_dir)
        run_output = output_dir / run_name
        run_output.mkdir(parents=True, exist_ok=True)
        try:
            if write_grid_vtk:
                _, grid_field, _, _ = data.read_step(step_start, color_field)
                _write_boutpp_grid(
                    data,
                    run_output / "boutpp-grid.vtm",
                    overwrite,
                    field_values=grid_field,
                    field_name=color_field,
                )
        except Exception:
            data.close()
            raise
        manifest_path = run_output / "manifest.csv"
        rows = []
        images = []
        background_mesh = None
        background_output = (
            Path(grid_output).expanduser().resolve()
            if grid_output is not None
            else run_output / "background.vtk"
        )
        stop = data.nsteps if step_stop is None else min(step_stop, data.nsteps)
        if step_start >= stop:
            data.close()
            raise ValueError(
                f"Empty timestep range for {run_name}: "
                f"start={step_start}, stop={stop}, available={data.nsteps}"
            )

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        background_actor = None
        filament_actor = None

        print(
            f"  {data.nsteps} available steps; rendering "
            f"range({step_start}, {stop}, {step_stride})",
            flush=True,
        )
        for step in range(step_start, stop, step_stride):
            row = dict(
                run=run_name,
                input_dir=str(data.run_dir),
                filename="",
                step=step,
                simulation_time="",
                iso_level="",
                iso_fraction="" if iso_fraction is None else iso_fraction,
                P_min="",
                P_max="",
                heatflux_par_e_min="",
                heatflux_par_e_max="",
                color_field=color_field,
                color_min="",
                color_max="",
                background_cells="",
                filament_cells="",
                status="",
                error="",
            )
            try:
                format_time = float("nan")
                filename = filename_pattern.format(
                    run=run_name, step=step, time=format_time
                )
                output_path = run_output / filename
                row["filename"] = str(output_path)
                if output_path.exists() and not overwrite:
                    row["status"] = "exists"
                    images.append(output_path)
                    print(f"  step {step}: exists, skipping {output_path}", flush=True)
                else:
                    pressure, color, heatflux, simulation_time = data.read_step(
                        step, color_field
                    )
                    if simulation_time is not None and "{time" in filename_pattern:
                        filename = filename_pattern.format(
                            run=run_name, step=step, time=simulation_time
                        )
                        output_path = run_output / filename
                        row["filename"] = str(output_path)

                    pressure_max = float(np.nanmax(pressure))
                    level = (
                        float(iso_value)
                        if iso_value is not None
                        else float(iso_fraction) * pressure_max
                    )
                    if level <= 0.0:
                        raise ValueError(
                            f"The selected positive P isovalue is {level:.12e}"
                        )
                    limits = color_limits or _automatic_color_limits(
                        color, color_field, color_percentile
                    )
                    if background_mesh is None:
                        background_mesh = _build_background_mesh(
                            data, color, color_field
                        )
                        background_output.parent.mkdir(parents=True, exist_ok=True)
                        background_mesh.save(str(background_output))
                        print(
                            f"  saved background grid {background_output}",
                            flush=True,
                        )
                    else:
                        _update_background_scalars(
                            data, background_mesh, color, color_field
                        )
                    (
                        background_cells,
                        filament_cells,
                        background_actor,
                        filament_actor,
                    ) = _render_frame(
                        data,
                        background_mesh,
                        pressure,
                        color,
                        color_field,
                        level,
                        simulation_time,
                        output_path,
                        limits,
                        window_size,
                        plotter,
                        background_actor,
                        filament_actor,
                    )
                    row.update(
                        simulation_time=(
                            "" if simulation_time is None else simulation_time
                        ),
                        iso_level=level,
                        P_min=float(np.nanmin(pressure)),
                        P_max=pressure_max,
                        heatflux_par_e_min=float(np.nanmin(heatflux)),
                        heatflux_par_e_max=float(np.nanmax(heatflux)),
                        color_min=limits[0],
                        color_max=limits[1],
                        background_cells=background_cells,
                        filament_cells=filament_cells,
                        status="rendered",
                    )
                    images.append(output_path)
                    print(
                        f"  step {step}: P iso={level:.6e}; saved {output_path}",
                        flush=True,
                    )
            except Exception as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                print(
                    f"  step {step}: {row['error']}", file=sys.stderr, flush=True
                )
                if not continue_on_error:
                    rows.append(row)
                    _write_manifest(manifest_path, rows)
                    plotter.close()
                    data.close()
                    raise
            rows.append(row)
            _write_manifest(manifest_path, rows)

        results[run_name] = images
        rendered = sum(row["status"] == "rendered" for row in rows)
        existing = sum(row["status"] == "exists" for row in rows)
        failed = sum(row["status"] == "error" for row in rows)
        print(
            f"Completed {run_name}: rendered={rendered}, existing={existing}, "
            f"failed={failed}; manifest={manifest_path}",
            flush=True,
        )
        plotter.close()
        data.close()
        del data
        gc.collect()
    return results


def _parse_named_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="[NAME=]DIR",
        help="Run directory; repeat for multiple runs",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--iso-fraction", type=float, default=0.50)
    parser.add_argument("--iso-value", type=float)
    parser.add_argument("--color-field", default="phi")
    parser.add_argument(
        "--write-grid-vtk",
        action="store_true",
        help="Write the VisIt-compatible BOUT++ grid to boutpp-grid.vtm",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    named_inputs = dict(_parse_named_input(value) for value in args.input)
    iso_fraction = None if args.iso_value is not None else args.iso_fraction
    render_filaments(
        named_inputs,
        args.output,
        step_start=args.start,
        step_stop=args.stop,
        step_stride=args.stride,
        iso_fraction=iso_fraction,
        iso_value=args.iso_value,
        color_field=args.color_field,
        write_grid_vtk=args.write_grid_vtk,
        overwrite=args.overwrite,
        continue_on_error=not args.fail_fast,
    )


if __name__ == "__main__":
    _main()
