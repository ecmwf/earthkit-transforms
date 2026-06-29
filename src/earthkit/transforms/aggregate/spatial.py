# Copyright 2024-, European Centre for Medium Range Weather Forecasts.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

import logging

from earthkit.transforms.spatial._aggregate import mask, reduce

from ._deprecate import _deprecated

logger = logging.getLogger(__name__)

mask = _deprecated(mask, new_module="earthkit.transforms.spatial")
reduce = _deprecated(reduce, new_module="earthkit.transforms.spatial")


def masks(
    *_args,
    **_kwargs,
):
    logger.warning(
        "earthkit.transforms.aggregate.spatial.masks is deprecated, "
        "please use earthkit.transforms.aggregate.spatial.mask instead."
    )
    return mask(*_args, **_kwargs)


__all__ = [mask, "masks", reduce]
