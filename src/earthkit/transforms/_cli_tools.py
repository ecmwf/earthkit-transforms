# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared helpers, arguments and options for the earthkit-transforms command line commands."""

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


io_arguments = [
    click.argument("how", metavar="HOW", type=click.STRING),
    click.argument("input_file", metavar="INPUT", type=click.Path(exists=True, dir_okay=False)),
    click.argument("output_file", metavar="OUTPUT", type=click.Path(dir_okay=False, writable=True)),
]

read_options = [
    click.option(
        "--profile",
        default=None,
        help="Name of the earthkit Xarray engine profile used when opening GRIB data, e.g. 'mars'. "
        "Uses the earthkit-data default if not given.",
    ),
]


def reduce_file(reduce_func, how, input_file, output_file, profile=None, **kwargs):
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
