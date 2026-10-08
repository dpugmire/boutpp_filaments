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
        help="Exclusive timestep stop; defaults to all available timesteps",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Timestep spacing (default: 1)",
    )
    parser.add_argument(
        "--iso-fraction",
        type=float,
        default=0.50,
        help="Pressure isovalue as a fraction of max(P), from 0 to 1 "
        "(default: 0.50)",
    )
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
        iso_fraction=args.iso_fraction,
        iso_range=None if args.iso_range is None else tuple(args.iso_range),
        trailing_steps=args.trailing_steps,
        overwrite=args.overwrite,
        grid_output=args.grid_output,
        write_grid_vtk=args.write_grid_vtk,
        write_merged_grid_vtk=args.write_merged_grid_vtk,
    )


if __name__ == "__main__":
    main()
