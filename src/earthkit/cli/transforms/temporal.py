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
from earthkit.cli.standard_args import (
    SOURCE_HELP,
    TARGET_HELP,
    add_options,
    profile_option,
    source_options,
    split_csv,
    target_options,
)

from earthkit.cli.transforms._tools import _reduce_file

_agg_options = [
    click.argument("how"),
    source_options(positional=True),
    target_options(positional=True),
    profile_option,
    click.option(
        "--time-dim",
        help="Name of the time dimension or coordinate. Deduced from the data by default.",
    ),
    click.option(
        "--time-shift",
        help="Time shift applied before the calculation, e.g. '3h' or '-30min', "
        "or the name of a coordinate holding per-gridpoint offsets.",
    ),
    click.option(
        "--extra-reduce-dims",
        multiple=True,
        callback=split_csv,
        help="Additional dimensions to reduce over, e.g. 'latitude,longitude'. Comma-separated or repeated.",
    ),
]

_AGG_HELP = """Aggregate data to {frequency} values.

HOW is the reduction applied to each {period}'s data, e.g. 'mean', 'max', 'min' or 'sum'.
Any xarray reduction method, earthkit-transforms method or numpy function name is accepted.

SOURCE: {source_help}

TARGET: {target_help}

\b
Example:
{examples}
"""


def _agg_command(frequency, period, *examples):
    """Return the ``earthkit {frequency}-agg`` command, wrapping ``earthkit.transforms.temporal.{frequency}_reduce``."""
    help = _AGG_HELP.format(
        frequency=frequency,
        period=period,
        source_help=SOURCE_HELP,
        target_help=TARGET_HELP,
        examples="\n".join(f"    earthkit {frequency}-agg {example}" for example in examples),
    )

    @earthkit.command(name=f"{frequency}-agg", help=help)
    @add_options(_agg_options)
    def command(**kwargs):
        from earthkit.transforms import temporal

        _reduce_file(getattr(temporal, f"{frequency}_reduce"), **kwargs)

    return command


daily_agg = _agg_command("daily", "day", "mean input.grib --time-shift 3h output.nc")
monthly_agg = _agg_command(
    "monthly",
    "month",
    "sum input.grib --extra-reduce-dims latitude,longitude output.nc",
    """mean \\
        'cds:{"dataset": "reanalysis-era5-single-levels", "variable": "2m_temperature", "year": "2020"}' \\
        output.nc""",
)
yearly_agg = _agg_command("yearly", "year", "max input.grib output.nc")
