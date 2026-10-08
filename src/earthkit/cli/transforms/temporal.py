# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Temporal commands of earthkit-transforms for the shared ``earthkit`` command line interface.

Provides ``earthkit daily-agg``, ``earthkit monthly-agg`` and ``earthkit yearly-agg``, which wrap the
reductions in :mod:`earthkit.transforms.temporal`.
"""

import click
from earthkit.cli.main import earthkit
from earthkit.cli.standard_args import add_options, split_csv

from earthkit.cli.transforms._tools import _io_arguments, _read_options, _reduce_file

_temporal_reduce_options = [
    click.option(
        "-t",
        "--time-dim",
        default=None,
        help="Name of the time dimension or coordinate. Deduced from the data by default.",
    ),
    click.option(
        "-s",
        "--time-shift",
        default=None,
        help="Time shift applied before the calculation, e.g. '3h' or '-30min', "
        "or the name of a coordinate holding per-gridpoint offsets.",
    ),
    click.option(
        "-r",
        "--extra-reduce-dims",
        multiple=True,
        callback=split_csv,
        help="Additional dimensions to reduce over, e.g. 'latitude,longitude'. Comma-separated or repeated.",
    ),
]


@earthkit.command(name="daily-agg")
@add_options(_io_arguments + _read_options + _temporal_reduce_options)
def daily_agg(**kwargs):
    """Aggregate data to daily values.

    HOW is the reduction applied to each day's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    The data is read from the earthkit-data source given by --source [NAME:]VALUE, e.g. a file (GRIB, NetCDF, ...),
    a URL or a JSON request to the CDS or MARS. The result is written to the earthkit-data target given by
    --target [NAME:]VALUE, e.g. a file or a Zarr store.

    \b
    Example:
        earthkit daily-agg mean --source input.grib --target output.nc --time-shift 3h
    """  # noqa: D301 (\b is a Click paragraph marker)
    from earthkit.transforms import temporal

    _reduce_file(temporal.daily_reduce, **kwargs)


@earthkit.command(name="monthly-agg")
@add_options(_io_arguments + _read_options + _temporal_reduce_options)
def monthly_agg(**kwargs):
    """Aggregate data to monthly values.

    HOW is the reduction applied to each month's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    The data is read from the earthkit-data source given by --source [NAME:]VALUE, e.g. a file (GRIB, NetCDF, ...),
    a URL or a JSON request to the CDS or MARS. The result is written to the earthkit-data target given by
    --target [NAME:]VALUE, e.g. a file or a Zarr store.

    \b
    Example:
        earthkit monthly-agg sum --source input.grib --target output.nc \\
            --extra-reduce-dims latitude,longitude
        earthkit monthly-agg mean --target file:output.nc --source \\
            'cds:{"dataset": "reanalysis-era5-single-levels", "variable": "2m_temperature", "year": "2020"}'
    """  # noqa: D301 (\b is a Click paragraph marker)
    from earthkit.transforms import temporal

    _reduce_file(temporal.monthly_reduce, **kwargs)


@earthkit.command(name="yearly-agg")
@add_options(_io_arguments + _read_options + _temporal_reduce_options)
def yearly_agg(**kwargs):
    """Aggregate data to yearly values.

    HOW is the reduction applied to each year's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    The data is read from the earthkit-data source given by --source [NAME:]VALUE, e.g. a file (GRIB, NetCDF, ...),
    a URL or a JSON request to the CDS or MARS. The result is written to the earthkit-data target given by
    --target [NAME:]VALUE, e.g. a file or a Zarr store.

    \b
    Example:
        earthkit yearly-agg max --source input.grib --target output.nc
    """  # noqa: D301 (\b is a Click paragraph marker)
    from earthkit.transforms import temporal

    _reduce_file(temporal.yearly_reduce, **kwargs)
