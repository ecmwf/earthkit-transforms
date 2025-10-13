<p align="center">
  <picture>
    <source srcset="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-transforms-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-transforms-light.svg" height="120">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/incubating_badge.svg" alt="Maturity Level">
  </a>
  <!-- <a href="https://codecov.io/gh/ecmwf/earthkit-data">
    <img src="https://codecov.io/gh/ecmwf/earthkit-data/branch/main/graph/badge.svg" alt="Code Coverage">
  </a> -->
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/earthkit-transforms/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-transforms?color=purple&label=Release" alt="Latest Release">
  </a>
  <!-- <a href="https://earthkit-data.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/earthkit-data/badge/?version=latest" alt="Documentation Status">
  </a> -->
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a>
  •
  <a href="#installation">Installation</a>
  •
  <a href="https://earthkit-transforms.readthedocs.io/en/latest/">Documentation</a>
</p>

> \[!IMPORTANT\]
> This software is **Incubating** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

**earthkit-transforms** is a library of software tools to support people working with climate and meteorology data.

**earthkit-transforms** is made up of sub-modules which contains methods for transforming data in specific domains.
For example, the temporal module contains methods for aggregation and statistics analysis accross time dimensions/coordinates.

It has been designed following the philosphy of [earthkit](https://github.com/ecmwf/earthkit), hence the methods should be interoperable with any
data object understood by earthkit-data.

## Installation

```bash
pip install earthkit-transforms
```

## Quick Start

```python
import earthkit.transforms as ekt

daily_mean = ekt.temporal.daily_mean(MY_DATA)

```

## Detailed documentation

Please refer the [earthkit-transforms readthedocs page](https://earthkit-transforms.readthedocs.io) for more detailed documentation, example notebooks and the API reference guide.

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with latest stable version of Python (version 3.13 at time of writing):

```
conda create -n DEVELOP -c conda-forge python=3.13
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make default`

## Licence

```
Copyright 2022, European Centre for Medium Range Weather Forecasts.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
