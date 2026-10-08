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
from earthkit.cli.standard_args import profile_option, source_options, target_options

_io_arguments = [
    click.argument("how", metavar="HOW", type=click.STRING),
    source_options(),
    target_options(),
]

_read_options = [profile_option]


def _reduce_file(reduce_func, how, source, target, profile=None, **kwargs):
    """Read SOURCE with earthkit-data, apply ``reduce_func`` and write the result to TARGET.

    ``source`` and ``target`` are the :class:`~earthkit.cli.standard_args.Source` and
    :class:`~earthkit.cli.standard_args.Target` built by :func:`~earthkit.cli.standard_args.source_options`
    and :func:`~earthkit.cli.standard_args.target_options`.
    """
    try:
        import earthkit.data  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "earthkit-data is required to read input data, install it with 'pip install earthkit-transforms[all]'"
        )

    # Fall back to the reduce function defaults, don't duplicate here
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    xarray_kwargs = {"profile": profile} if profile is not None else {}

    in_data = source.from_source().to_xarray(**xarray_kwargs)
    out_data = reduce_func(in_data, how=how, **kwargs)
    target.to_target(out_data)
