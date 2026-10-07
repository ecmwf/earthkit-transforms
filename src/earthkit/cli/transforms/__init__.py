# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Commands contributed by earthkit-transforms to the shared ``earthkit`` command line interface.

The ``earthkit`` console script itself lives in :mod:`earthkit.cli.main` (earthkit-utils). This package is
part of the ``earthkit.cli`` namespace package, which is shared by all earthkit packages. Importing it
imports its submodules, which register their commands on the shared ``earthkit`` group with
``@earthkit.command()``. The submodules mirror those of :mod:`earthkit.transforms`, e.g.
:mod:`earthkit.cli.transforms.temporal` provides ``earthkit daily-agg``, ``earthkit monthly-agg`` and
``earthkit yearly-agg``. To add commands for another submodule, create ``earthkit/cli/transforms/<name>.py``
and import it below.

This package lives outside of ``earthkit.transforms`` on purpose, so that listing the commands does not import
``earthkit.transforms``. Only import :mod:`click` and light standard library modules at module level, and
import everything else inside the command functions.
"""

from earthkit.cli.transforms import temporal  # noqa: F401
