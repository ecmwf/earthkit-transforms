# Welcome to Earthkit-transforms's documentation!

**earthkit-transforms** is a library of software tools to support people working with climate and meteorology data

**earthkit-transforms** includes methods for aggregating data in time and space, and more transforms and operators.
It has been designed following the philosphy of Earthkit, hence the methods should be interoperable with any
data object understood by earthkit-data.

## Getting started

**earthkit-transforms** is available from PyPi. To make use of the interoperable functionality you should ensure
that you have installed the dependancies of *earthkit-data*.

### Installation

```bash
pip install earthkit-transforms
```

### Import and use

```python
from earthkit.transforms import aggregate

daily_mean = aggregate.temporal.daily_mean(MY_DATA)

```

```{toctree}
:caption: 'Examples'
:maxdepth: 2

aggregate-examples.md

:caption: 'Documentation'
:maxdepth: 2
install.md
API Reference <_api/index>


```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
