# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers, arguments and options shared by the earthkit-transforms commands.

Arguments and options common to all earthkit packages come from :mod:`earthkit.cli.standard_args`.

Only import :mod:`click` and light standard library modules at module level, see
:mod:`earthkit.cli.transforms`.
"""

import click
from earthkit.cli.standard_args import profile_option, source_file_argument, target_file_argument

_io_arguments = [
    click.argument("how", metavar="HOW", type=click.STRING),
    source_file_argument,
    target_file_argument,
]

_read_options = [profile_option]


def _reduce_file(reduce_func, how, source_file, target_file, profile=None, **kwargs):
    """Read SOURCE_FILE with earthkit-data, apply ``reduce_func`` and write the result to TARGET_FILE."""
    try:
        import earthkit.data as ekd
    except ImportError:
        raise click.ClickException(
            "earthkit-data is required to read input files, install it with 'pip install earthkit-transforms[all]'"
        )

    # Fall back to the reduce function defaults, don't duplicate here
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    xarray_kwargs = {"profile": profile} if profile is not None else {}

    in_data = ekd.from_source("file", source_file).to_xarray(**xarray_kwargs)
    out_data = reduce_func(in_data, how=how, **kwargs)
    ekd.to_target("file", target_file, data=out_data)
