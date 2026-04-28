# earthkit-transforms

A library of quality-assured methods for transforming climate and meteorological data,
built on xarray. Developed and maintained by ECMWF.

- **Docs**: https://earthkit-transforms.readthedocs.io/en/latest/
- **Repo**: https://github.com/ecmwf/earthkit-transforms

---

## Architecture

Primary import: `import earthkit.transforms as ekt`

| Sub-module | Purpose |
|---|---|
| `ekt.temporal` | Time aggregations, rolling reductions, rate/deaccumulation |
| `ekt.climatology` | Climatological statistics (groupby year), anomalies |
| `ekt.spatial` | Spatial masking and aggregation using geometries |
| `ekt.ensemble` | Reduction over ensemble member dimensions |
| `ekt.aggregate` | **Deprecated** — will be removed in v2.x; use domain modules above |

Top-level convenience functions: `ekt.reduce`, `ekt.resample`, `ekt.rolling_reduce`.

All functions accept `xr.DataArray` or `xr.Dataset`. They are also compatible with
earthkit-data objects directly (converted internally by the `format_handler` decorator).

Source lives in `src/earthkit/transforms/`. Each sub-module has a domain `_aggregate.py`
(and `_rates.py` for temporal). Shared utilities are in `src/earthkit/transforms/_tools.py`.

---

## API Conventions

### `how` parameter

Specifies the reduction method. Accepted values:

| String | Equivalent |
|---|---|
| `"mean"`, `"average"`, `"nanaverage"` | NaN-aware mean (or weighted mean) |
| `"median"` | NaN-aware median |
| `"min"`, `"max"` | NaN-aware min/max |
| `"sum"` | NaN-aware sum |
| `"std"`, `"stddev"`, `"stdev"` | NaN-aware standard deviation |
| `"q"`, `"quantile"` | Quantile |
| `"percentile"`, `"p"` | Percentile |

`how` also accepts a callable, or a fully-qualified string like `"numpy.nanmean"` or
`"cupy.nanmean"`. Method resolution order: xarray built-ins → earthkit aliases → array namespace.

### `frequency` parameter (temporal / climatology)

Both pandas offset aliases and xarray time-component strings are accepted:

| Input | Resolves to |
|---|---|
| `"D"` | `"dayofyear"` |
| `"W"` | `"weekofyear"` |
| `"M"`, `"ME"`, `"MS"` | `"month"` |
| `"H"` | `"hour"` |
| `"YE"` | `"year"` |
| `"season"` | `"season"` (DJF/MAM/JJA/SON) |
| `"dayofyear"`, `"weekofyear"`, `"month"`, `"year"`, `"hour"` | passed through directly |

**Climatology-only restriction**: `"day"` is invalid — use `"dayofyear"`.

When `frequency` is passed to `temporal.reduce`, the function delegates to `resample` (not `groupby`).

### Dimension detection

Dimensions are auto-detected in this priority order:
1. `axis` attribute on the coordinate (CF: `"T"`, `"X"`, `"Y"`, `"E"`)
2. `standard_name` attribute (CF conventions)
3. Common variable names (see table below)

| Axis | Recognised names |
|---|---|
| Time | `time`, `valid_time`, `forecast_reference_time` |
| Latitude | `lat`, `latitude` |
| Longitude | `lon`, `long`, `longitude` |
| Ensemble | `ensemble_member`, `ensemble`, `member`, `number`, `realization`, `realisation` |

Override with: `time_dim=`, `lat_key=`, `lon_key=`, or `dim=` as appropriate.
CF-compliant data gives the most reliable results.

### Weighting

Pass `weights="latitude"` (or `"lat"`) to apply cosine-latitude weighting.
Custom weight arrays are also accepted. Only available with weighted-compatible `how` methods.

### `time_shift` parameter

Available on temporal functions. Accepts `dict`, `str`, or `pd.Timedelta`:

```python
temporal.reduce(data, time_shift={"hours": 12})
temporal.reduce(data, time_shift="12h")
```

Use `remove_partial_periods=True` to drop the boundary bins introduced by the shift.

### `how_label` parameter

Appends a suffix to output variable names:

```python
temporal.reduce(data, how="mean", how_label="mean")  # variable becomes "2t_mean"
```

---

## Common Patterns

```python
import earthkit.transforms as ekt

# Temporal reduction — collapse entire time dimension
ekt.temporal.mean(ds)
ekt.temporal.reduce(ds, how="max")

# Resample to daily/monthly (pass frequency)
ekt.temporal.reduce(ds, frequency="D", how="mean")    # daily mean
ekt.temporal.daily_mean(ds)                            # equivalent shorthand
ekt.temporal.reduce(ds, frequency="ME", how="sum")    # monthly sum
ekt.temporal.monthly_sum(ds)                           # equivalent shorthand

# Rolling window
ekt.temporal.rolling_reduce(ds, time=7, how_reduce="mean")

# Deaccumulation / rate conversion
ekt.temporal.deaccumulate(ds)                          # step-length rate (e.g. seasonal forecasts)
ekt.temporal.accumulation_to_rate(ds, rate_units="hours", accumulation_type="start_of_step")

# Climatology (group across years, reduce within each group)
ekt.climatology.mean(ds, frequency="month")
ekt.climatology.reduce(ds, frequency="dayofyear", how="std", climatology_range=(1991, 2020))

# Anomaly
ekt.climatology.anomaly(data, reference_climatology)
ekt.climatology.auto_anomaly(ds, frequency="month", climatology_range=(1991, 2020))

# Spatial (requires geopandas geometry object)
ekt.spatial.mask(ds, geometry)
ekt.spatial.reduce(ds, geometry, how="mean")

# Ensemble
ekt.ensemble.mean(ds)
ekt.ensemble.reduce(ds, how="std")
```

---

## Dependencies

| Package | When required |
|---|---|
| `xarray`, `numpy`, `earthkit-utils`, `geopandas` | Always |
| `earthkit-data` | For reading GRIB/NetCDF sources; install with `pip install earthkit-transforms[all]` |
| `rasterio` | For rasterized spatial masking on regular grids; install with `pip install earthkit-transforms[all]` |

Python 3.10+ required.

---

## Build and Test (contributors)

```bash
# Set up development environment (choose one)
make clean-pip-env && . venv/bin/activate
make clean-uv-env  && . .venv/bin/activate
make clean-conda-env && conda activate ./.conda

# Install with CI extras
pip install -e ".[ci]"

# Run everything (QA + unit tests)
make default

# Run individually
make qa                       # pre-commit (ruff, mypy, etc.)
make unit-tests               # pytest tests/ (excludes legacy-api/)
make legacy-api-unit-tests    # pytest tests/legacy-api/
make type-check               # mypy
make docs-build               # Sphinx HTML docs
```

Tests fetch remote ERA5 test data (cached after first run via earthkit-data cache).
Legacy API tests cover the deprecated `earthkit.transforms.aggregate` namespace.

---

## Conventions (contributors)

- New sub-module functions belong in the domain's `_aggregate.py` (or `_rates.py` for temporal rates).
- Decorate public functions with `@format_handler()` (from `earthkit.utils.decorators`) so they accept
  earthkit-data objects directly. Apply `@_tools.time_dim_decorator` on any function that operates on
  a time dimension.
- All public functions must handle both `xr.DataArray` and `xr.Dataset` — iterate over `data_vars`
  for Datasets (see `_aggregate.py` patterns).
- Add the function to the sub-module's `__init__.py` `__all__` list.
- Convenience wrappers (e.g. `daily_mean`) should just set `how` and delegate to `reduce`.
- Do not add functions to the deprecated `aggregate/` namespace.
- See [contributing guide](docs/source/contributing.rst) for environment setup and CI details.
