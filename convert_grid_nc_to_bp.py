#!/usr/bin/env python3
"""Convert a BOUT++ NetCDF grid file to a static ADIOS BP file.

The output contains one ADIOS step. Every numeric NetCDF variable is copied
with its original name, shape, and element type. NetCDF dimensions and
attributes are preserved as ADIOS attributes.

Example
-------

    python convert_grid_nc_to_bp.py grid.nc grid.bp
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import adios2
import numpy as np
from scipy.io import netcdf_file


DIMENSIONS_ATTRIBUTE = "netcdf/dimensions"
VARIABLE_METADATA_PREFIX = "netcdf/variables"
GLOBAL_ATTRIBUTE_PREFIX = "netcdf/global"


def _native_array(value: Any) -> np.ndarray:
    """Return a contiguous array with native byte order."""
    array = np.asarray(value)
    if array.dtype.byteorder not in ("=", "|") and not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    return np.ascontiguousarray(array)


def _attribute_value(value: Any) -> Any:
    """Convert a NetCDF attribute to a value accepted by ADIOS2."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="surrogateescape")
    if isinstance(value, str):
        return value

    array = np.asarray(value)
    if array.dtype.kind == "S":
        strings = [
            item.decode("utf-8", errors="surrogateescape")
            for item in array.reshape(-1).tolist()
        ]
        return strings[0] if array.ndim == 0 else strings
    if array.dtype.kind == "U":
        strings = array.reshape(-1).tolist()
        return strings[0] if array.ndim == 0 else strings
    if array.dtype.kind == "O":
        raise TypeError(f"Unsupported object-valued NetCDF attribute: {value!r}")

    array = _native_array(array)
    return array.item() if array.ndim == 0 else array


def _load_netcdf(path: Path) -> tuple[dict, dict[str, dict], dict]:
    """Read a NetCDF grid completely so the file can be closed before writing."""
    variables: dict[str, dict] = {}
    with netcdf_file(str(path), "r", mmap=False) as grid:
        dimensions = {
            name: None if size is None else int(size)
            for name, size in grid.dimensions.items()
        }
        global_attributes = dict(grid._attributes)

        for name, variable in grid.variables.items():
            values = _native_array(np.array(variable.data, copy=True))
            if values.dtype.kind not in "biufc":
                raise TypeError(
                    f"Variable {name!r} has unsupported non-numeric dtype "
                    f"{values.dtype}"
                )
            variables[name] = {
                "values": values,
                "dimensions": tuple(variable.dimensions),
                "attributes": dict(variable._attributes),
                "netcdf_dtype": np.asarray(variable.data).dtype.str,
            }
    return dimensions, variables, global_attributes


def _remove_output(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _write_attribute(
    stream: adios2.Stream,
    name: str,
    value: Any,
    *,
    variable_name: str = "",
) -> None:
    stream.write_attribute(
        name,
        _attribute_value(value),
        variable_name=variable_name,
        separator="/",
    )


def _write_bp(
    path: Path,
    dimensions: dict,
    variables: dict[str, dict],
    global_attributes: dict,
) -> None:
    with adios2.Stream(str(path), "w") as stream:
        _write_attribute(
            stream,
            DIMENSIONS_ATTRIBUTE,
            json.dumps(dimensions, sort_keys=True),
        )
        _write_attribute(stream, "netcdf/source_format", "NetCDF")

        for name, value in global_attributes.items():
            _write_attribute(
                stream,
                f"{GLOBAL_ATTRIBUTE_PREFIX}/{name}",
                value,
            )

        adios_variables = {}
        for name, metadata in variables.items():
            values = metadata["values"]
            shape = list(values.shape)
            start = [0] * values.ndim
            count = list(values.shape)
            adios_variables[name] = stream.io.define_variable(
                name,
                values,
                shape,
                start,
                count,
                True,
            )
            _write_attribute(
                stream,
                f"{VARIABLE_METADATA_PREFIX}/{name}/dimensions",
                json.dumps(metadata["dimensions"]),
            )
            _write_attribute(
                stream,
                f"{VARIABLE_METADATA_PREFIX}/{name}/netcdf_dtype",
                metadata["netcdf_dtype"],
            )
            for attribute_name, attribute_value in metadata["attributes"].items():
                _write_attribute(
                    stream,
                    attribute_name,
                    attribute_value,
                    variable_name=name,
                )

        stream.begin_step()
        for name, variable in adios_variables.items():
            stream.write(variable, variables[name]["values"])
        stream.end_step()


def _attribute_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar string attribute, got {array.shape}")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="surrogateescape")
    return str(item)


def verify_conversion(
    bp_path: str | Path,
    dimensions: dict,
    variables: dict[str, dict],
) -> None:
    """Reopen a converted BP file and compare every array to its source."""
    bp_path = Path(bp_path)
    with adios2.FileReader(str(bp_path)) as reader:
        available_variables = reader.available_variables()
        expected_names = set(variables)
        actual_names = set(available_variables)
        if expected_names != actual_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"Variable mismatch: missing={missing}, unexpected={extra}"
            )

        for name, metadata in variables.items():
            expected = metadata["values"]
            actual = np.asarray(reader.read(name))
            if actual.shape != expected.shape:
                raise ValueError(
                    f"{name}: expected shape {expected.shape}, got {actual.shape}"
                )
            if (
                actual.dtype.kind != expected.dtype.kind
                or actual.dtype.itemsize != expected.dtype.itemsize
            ):
                raise ValueError(
                    f"{name}: expected dtype equivalent to {expected.dtype}, "
                    f"got {actual.dtype}"
                )
            if not np.array_equal(actual, expected, equal_nan=True):
                raise ValueError(f"{name}: data values differ after conversion")

        stored_dimensions = json.loads(
            _attribute_text(reader.read_attribute(DIMENSIONS_ATTRIBUTE))
        )
        if stored_dimensions != dimensions:
            raise ValueError(
                f"Dimension metadata differs: expected {dimensions}, "
                f"got {stored_dimensions}"
            )


def convert_grid(
    netcdf_path: str | Path,
    bp_path: str | Path,
    *,
    overwrite: bool = False,
    verify: bool = True,
) -> dict:
    """Convert one BOUT++ NetCDF grid into a one-step ADIOS BP dataset."""
    netcdf_path = Path(netcdf_path).expanduser().resolve()
    bp_path = Path(bp_path).expanduser().resolve()

    if not netcdf_path.is_file():
        raise FileNotFoundError(f"NetCDF input does not exist: {netcdf_path}")
    if netcdf_path == bp_path:
        raise ValueError("Input and output paths must differ")
    if bp_path.exists() or bp_path.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {bp_path}; use --overwrite to replace it"
            )
        _remove_output(bp_path)
    bp_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions, variables, global_attributes = _load_netcdf(netcdf_path)
    _write_bp(bp_path, dimensions, variables, global_attributes)
    if verify:
        verify_conversion(bp_path, dimensions, variables)

    total_bytes = sum(
        metadata["values"].nbytes for metadata in variables.values()
    )
    return {
        "input": netcdf_path,
        "output": bp_path,
        "variables": len(variables),
        "dimensions": dimensions,
        "uncompressed_bytes": total_bytes,
        "verified": verify,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("netcdf_path", type=Path, help="Input grid.nc file")
    parser.add_argument("bp_path", type=Path, help="Output grid.bp path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file or BP directory",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not reopen and compare the output after writing",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    summary = convert_grid(
        args.netcdf_path,
        args.bp_path,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )
    print(f"Input: {summary['input']}")
    print(f"Output: {summary['output']}")
    print(f"Variables: {summary['variables']}")
    print(f"Dimensions: {summary['dimensions']}")
    print(f"Uncompressed data: {summary['uncompressed_bytes']:,} bytes")
    print(f"Verified: {summary['verified']}")


if __name__ == "__main__":
    _main()
