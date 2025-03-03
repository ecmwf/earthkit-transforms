<h3>
<picture>
    <source srcset="https://raw.githubusercontent.com/ecmwf/logos/refs/heads/main/logos/earthkit/earthkit-transforms-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/ecmwf/logos/refs/heads/main/logos/earthkit/earthkit-transforms-light.svg" width="300">
</picture>
</h3>

<p>
  <img src="https://img.shields.io/badge/ESEE-Foundation-orange" alt="ESEE Foundation">
  <a href="https://github.com/ecmwf/codex/blob/cookiecutter/Project%20Maturity/project-maturity.md">
    <img src="https://img.shields.io/badge/Maturity-Incubating-lightskyblue" alt="Maturity Incubating">
  </a>
</p><p>
  <a href="https://github.com/ecmwf/earthkit-transforms/actions/workflows/on-push.yaml">
    <img src="https://github.com/ecmwf/earthkit-transforms/actions/workflows/on-push.yaml/badge.svg" alt="CI Status">
  </a>
  <a href="https://github.com/ecmwf/earthkit-transforms/actions/workflows/on-release.yaml">
    <img src="https://github.com/ecmwf/earthkit-transforms/actions/workflows/on-release.yaml/badge.svg" alt="CI Status">
  </a>
  <a href="https://codecov.io/gh/ecmwf/earthkit-transforms">
    <img src="https://codecov.io/gh/ecmwf/earthkit-transforms/branch/develop/graph/badge.svg" alt="Code Coverage" >
  </a>
</p><p>
  <a href="https://github.com/ecmwf/earthkit-transforms/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-transforms?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://earthkit-transforms.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/earthkit-transforms/badge/?version=latest" alt="Documentation Status">
  </a>
</p>

# earthkit-transforms

**earthkit-transforms** is a library of software tools to support people working with climate and meteorology data.

**earthkit-transforms** is made up of a sub-package, `aggregate` which contains methods for aggregating data in time and space, and more transforms and operators.
It has been designed following the philosphy of Earthkit, hence the methods should be interoperable with any
data object understood by earthkit-data.

## Quick Start

### Installation

```bash
pip install earthkit-transforms
```

### Import and use

```python
from earthkit.transforms import aggregate

daily_mean = aggregate.temporal.daily_mean(MY_DATA)

```

## Detailed documentation

Please refer the [earthkit-transforms readthedocs page](https://earthkit-transforms.readthedocs.io) for more detailed documentation, example notebooks and the API reference guide.

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.11:

```
conda create -n DEVELOP -c conda-forge python=3.11
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make default`

## License

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
```
