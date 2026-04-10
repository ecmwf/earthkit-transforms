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

"""Temporary deprecated sub-module namespace."""

from earthkit.transforms._aggregate import reduce, resample, rolling_reduce

from . import climatology, ensemble, general, spatial, temporal

__all__ = [
    "climatology",
    "ensemble",
    "general",
    "spatial",
    "temporal",
    "reduce",
    "resample",
    "rolling_reduce",
]
