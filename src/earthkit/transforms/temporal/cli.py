# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Temporal commands for the ``earthkit`` command line interface.

The commands are collated with those of the other earthkit-transforms submodules in
:mod:`earthkit.transforms.cli`.
"""

import click

from earthkit.transforms import temporal
from earthkit.transforms._cli_tools import (
    _split_csv,
    add_options,
    io_arguments,
    read_options,
    reduce_file,
)

extra_reduce_dims_option = click.option(
    "-r",
    "--extra-reduce-dims",
    multiple=True,
    callback=_split_csv,
    help="Additional dimensions to reduce over, e.g. 'latitude,longitude'. Comma-separated or repeated.",
)


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
    extra_reduce_dims_option,
]


@click.command(name="daily-agg")
@add_options(io_arguments + read_options + _temporal_reduce_options)
def daily_agg(**kwargs):
    """Aggregate INPUT to daily values and write the result to OUTPUT as NetCDF.

    HOW is the reduction applied to each day's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    INPUT is any file readable by earthkit-data (e.g. GRIB or NetCDF).

    \b
    Example:
        earthkit daily-agg mean input.grib output.nc --time-shift 3h
    """  # noqa: D301 (\b is a Click paragraph marker)
    reduce_file(temporal.daily_reduce, **kwargs)


@click.command(name="monthly-agg")
@add_options(io_arguments + read_options + _temporal_reduce_options)
def monthly_agg(**kwargs):
    """Aggregate INPUT to monthly values and write the result to OUTPUT as NetCDF.

    HOW is the reduction applied to each month's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    INPUT is any file readable by earthkit-data (e.g. GRIB or NetCDF).

    \b
    Example:
        earthkit monthly-agg sum input.grib output.nc --extra-reduce-dims latitude,longitude
    """  # noqa: D301 (\b is a Click paragraph marker)
    reduce_file(temporal.monthly_reduce, **kwargs)


@click.command(name="yearly-agg")
@add_options(io_arguments + read_options + _temporal_reduce_options)
def yearly_agg(**kwargs):
    """Aggregate INPUT to yearly values and write the result to OUTPUT as NetCDF.

    HOW is the reduction applied to each year's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    INPUT is any file readable by earthkit-data (e.g. GRIB or NetCDF).

    \b
    Example:
        earthkit yearly-agg max input.grib output.nc
    """  # noqa: D301 (\b is a Click paragraph marker)
    reduce_file(temporal.yearly_reduce, **kwargs)


COMMANDS = {
    "daily-agg": daily_agg,
    "monthly-agg": monthly_agg,
    "yearly-agg": yearly_agg,
}
