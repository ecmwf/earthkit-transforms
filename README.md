# earthkit-transforms

**DISCLAIMER**

> This project is in the **BETA** stage of development. Please be aware that interfaces and functionality may change as the project develops. If this software is to be used in operational systems you are **strongly advised to use a released tag in your system configuration**, and you should be willing to accept incoming changes and bug fixes that require adaptations on your part. ECMWF **does use** this software in operations and abides by the same caveats.

**earthkit-transforms** is a library of software tools to support people working with climate and meteorology data

**earthkit-transforms** includes methods for aggregating data in time and space, and more transforms and operators.
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
