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

from earthkit.transforms import _aggregate as _src

from ._deprecate import _deprecated

# Explicitly wrap each function with a deprecation warning
how_label_rename = _deprecated(_src.how_label_rename, new_module="earthkit.transforms")
reduce = _deprecated(_src.reduce, new_module="earthkit.transforms")
rolling_reduce = _deprecated(_src.rolling_reduce, new_module="earthkit.transforms")
resample = _deprecated(_src.resample, new_module="earthkit.transforms")

# Explicitly declare __all__ for clean exports
__all__ = ["how_label_rename", "reduce", "rolling_reduce", "resample"]
