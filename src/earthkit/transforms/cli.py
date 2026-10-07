# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Commands contributed by earthkit-transforms to the shared ``earthkit`` command line interface.

The ``earthkit`` console script itself lives in :mod:`earthkit.utils.cli`, which discovers the
commands defined here through the ``COMMANDS`` mapping at the bottom of this module, so
``earthkit daily-agg <how> <input> <output>`` and ``earthkit monthly-agg <how> <input> <output>``
become available once earthkit-transforms is installed.
"""

import click


def _split_csv(ctx, param, value):
    """Turn a tuple of (possibly comma-separated) option values into a flat list."""
    if not value:
        return None
    result = []
    for item in value:
        result.extend(v.strip() for v in item.split(",") if v.strip())
    return result or None


# Function for grouping options
def add_options(options):
    """Apply a list of click arguments/options to a command, in the order they are listed."""

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


_io_arguments = [
    click.argument("how", metavar="HOW", type=click.STRING),
    click.argument("input_file", metavar="INPUT", type=click.Path(exists=True, dir_okay=False)),
    click.argument("output_file", metavar="OUTPUT", type=click.Path(dir_okay=False, writable=True)),
]

_read_options = [
    click.option(
        "--profile",
        default=None,
        help="Name of the earthkit Xarray engine profile used when opening GRIB data, e.g. 'mars'. "
        "Uses the earthkit-data default if not given.",
    ),
]

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
        callback=_split_csv,
        help="Additional dimensions to reduce over, e.g. 'latitude,longitude'. Comma-separated or repeated.",
    ),
]


def _reduce_file(reduce_func, how, input_file, output_file, profile=None, **kwargs):
    """Read INPUT with earthkit-data, apply ``reduce_func`` and write the result to OUTPUT."""
    try:
        import earthkit.data as ekd
    except ImportError:
        raise click.ClickException(
            "earthkit-data is required to read input files, install it with 'pip install earthkit-transforms[all]'"
        )

    # Fall back to the reduce function defaults, don't duplicate here
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    xarray_kwargs = {"profile": profile} if profile is not None else {}

    in_data = ekd.from_source("file", input_file).to_xarray(**xarray_kwargs)
    out_data = reduce_func(in_data, how=how, **kwargs)
    ekd.to_target("file", output_file, data=out_data)


@click.command(name="daily-agg")
@add_options(_io_arguments + _read_options + _temporal_reduce_options)
def daily_agg(**kwargs):
    """Aggregate INPUT to daily values and write the result to OUTPUT as NetCDF.

    HOW is the reduction applied to each day's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    INPUT is any file readable by earthkit-data (e.g. GRIB or NetCDF).

    \b
    Example:
        earthkit daily-agg mean input.grib output.nc --time-shift 3h
    """  # noqa: D301 (\b is a Click paragraph marker)
    from earthkit.transforms import temporal

    _reduce_file(temporal.daily_reduce, **kwargs)


@click.command(name="monthly-agg")
@add_options(_io_arguments + _read_options + _temporal_reduce_options)
def monthly_agg(**kwargs):
    """Aggregate INPUT to monthly values and write the result to OUTPUT as NetCDF.

    HOW is the reduction applied to each month's data, e.g. 'mean', 'max', 'min' or 'sum'.
    Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

    INPUT is any file readable by earthkit-data (e.g. GRIB or NetCDF).

    \b
    Example:
        earthkit monthly-agg sum input.grib output.nc --extra-reduce-dims latitude,longitude
    """  # noqa: D301 (\b is a Click paragraph marker)
    from earthkit.transforms import temporal

    _reduce_file(temporal.monthly_reduce, **kwargs)


COMMANDS = {
    "daily-agg": daily_agg,
    "monthly-agg": monthly_agg,
}
