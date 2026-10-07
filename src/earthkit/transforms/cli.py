# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Commands contributed by earthkit-transforms to the shared ``earthkit`` command line interface.

The ``earthkit`` console script itself lives in :mod:`earthkit.utils.cli`, which discovers the
commands collated here in the ``COMMANDS`` mapping. Each submodule defines its commands in its own
``cli`` module, e.g. :mod:`earthkit.transforms.temporal.cli` provides ``earthkit daily-agg``,
``earthkit monthly-agg`` and ``earthkit yearly-agg``.
"""

from earthkit.transforms.temporal import cli as temporal_cli

COMMANDS = {
    **temporal_cli.COMMANDS,
}
