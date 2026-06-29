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

"""Climatological transformations for earthkit data objects.

Typically this is done with an xarray representation of data.
"""

from ._aggregate import (
    anomaly,
    auto_anomaly,
    daily_max,
    daily_mean,
    daily_median,
    daily_min,
    daily_reduce,
    daily_std,
    max,
    mean,
    median,
    min,
    monthly_max,
    monthly_mean,
    monthly_median,
    monthly_min,
    monthly_reduce,
    monthly_std,
    percentiles,
    quantiles,
    reduce,
    relative_anomaly,
    std,
)

__all__ = [
    "reduce",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "daily_reduce",
    "daily_mean",
    "daily_median",
    "daily_min",
    "daily_max",
    "daily_std",
    "monthly_reduce",
    "monthly_mean",
    "monthly_median",
    "monthly_min",
    "monthly_max",
    "monthly_std",
    "quantiles",
    "percentiles",
    "anomaly",
    "relative_anomaly",
    "auto_anomaly",
]
