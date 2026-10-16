#!/usr/bin/env python3
"""Render a small selection of frames from the ELMO filament run."""

from __future__ import annotations

import argparse
from pathlib import Path

from filament_renderer import render_filaments


DEFAULT_INPUT = Path("/Users/dpn/proj/bout++/nersc_data/elmo")
DEFAULT_OUTPUT = DEFAULT_INPUT / "frames"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"ELMO run directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Frame output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        help="VTK path for the generated background grid; defaults to "
        "<output>/elmo/background.vtk",
    )
    parser.add_argument(
        "--write-grid-vtk",
        action="store_true",
        help="Write the complete VisIt-compatible BOUT++ grid as boutpp-grid.vtm",
    )
    parser.add_argument(
        "--write-merged-grid-vtk",
        action="store_true",
        help="Write one point-welded BOUT++ grid as boutpp-grid.vtu",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First timestep to render (default: 0)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Exclusive stop; if omitted, skips the final --trailing-steps (default: 10)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Timestep spacing (default: 1)",
    )
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument(
        "--iso-fraction",
        type=float,
        default=0.50,
        help="Pressure isovalue as a fraction of the range from min(P) to "
        "max(P), from 0 to 1 (default: 0.50)",
    )
    thresholds.add_argument("--iso-value", type=float, help="Absolute pressure threshold in code units")
    thresholds.add_argument("--iso-kpa", type=float, help="Common absolute physical pressure threshold in kPa")
    parser.add_argument(
        "--pressure-quantity", choices=["P", "fluctuation", "total"], default="P",
        help="Saved P, nonaxisymmetric P-mean_z(P), or P0+P (default: P)",
    )
    parser.add_argument("--grid-path", type=Path, help="Matching grid.bp path")
    parser.add_argument(
        "--grid-confirmed", action="store_true",
        help="Use a confirmed original input grid after shape/topology checks; "
        "record numerical comparisons separately",
    )
    parser.add_argument("--color-field", default="phi")
    parser.add_argument("--color-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--iso-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        help="Inclusive timestep range used for the iso-fraction pressure min/max",
    )
    parser.add_argument(
        "--trailing-steps",
        type=int,
        default=10,
        help="Final timesteps to skip when --stop is omitted (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace frames that already exist",
    )
    args = parser.parse_args()

    render_filaments(
        {"elmo": args.input},
        args.output,
        step_start=args.start,
        step_stop=args.stop,
        step_stride=args.stride,
        iso_fraction=None if args.iso_value is not None else args.iso_fraction,
        iso_value=args.iso_value,
        iso_kpa=args.iso_kpa,
        pressure_quantity=args.pressure_quantity,
        grid_path=args.grid_path,
        grid_confirmed=args.grid_confirmed,
        color_field=args.color_field,
        color_limits=None if args.color_limits is None else tuple(args.color_limits),
        iso_range=None if args.iso_range is None else tuple(args.iso_range),
        trailing_steps=args.trailing_steps,
        overwrite=args.overwrite,
        grid_output=args.grid_output,
        write_grid_vtk=args.write_grid_vtk,
        write_merged_grid_vtk=args.write_merged_grid_vtk,
    )


if __name__ == "__main__":
    main()
