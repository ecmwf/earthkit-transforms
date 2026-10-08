import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner
from earthkit.cli.main import earthkit

from earthkit.cli.transforms import temporal as temporal_cli

pytest.importorskip("earthkit.data")


@pytest.fixture
def netcdf_file(tmp_path):
    """Two years of daily data (2019 + leap year 2020) on a 2x3 grid, written to NetCDF."""
    time = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    data = np.arange(float(time.size * 6)).reshape(time.size, 2, 3)
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": [10.0, 20.0], "longitude": [0.0, 1.0, 2.0]},
    )
    path = tmp_path / "in.nc"
    ds.to_netcdf(path)
    return path, ds


def _io(how, source, target, *options):
    return [how, source, *options, target]


def _invoke(command, *args):
    result = CliRunner().invoke(command, [str(a) for a in args])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


@pytest.mark.parametrize(
    "name, command",
    (
        ("daily-agg", temporal_cli.daily_agg),
        ("monthly-agg", temporal_cli.monthly_agg),
        ("yearly-agg", temporal_cli.yearly_agg),
    ),
)
def test_cli_registers_commands(name, command):
    assert earthkit.get_command(None, name) is command


def test_cli_info_lists_transforms_commands():
    result = _invoke(earthkit, "info")
    assert "earthkit-transforms" in result.output
    assert "daily-agg, monthly-agg, yearly-agg" in result.output


@pytest.mark.parametrize("name", ("daily-agg", "monthly-agg", "yearly-agg"))
def test_cli_help(name):
    result = _invoke(earthkit, name, "--help")
    assert "[OPTIONS] HOW SOURCE TARGET\n" in result.output
    for text in ("--source", "--target"):
        assert text not in result.output
    for option in (
        "--profile",
        "--time-dim",
        "--time-shift",
        "--extra-reduce-dims",
    ):
        assert option in result.output


@pytest.mark.parametrize(
    "name, expected_length",
    (
        ("daily-agg", 731),
        ("monthly-agg", 24),
        ("yearly-agg", 2),
    ),
)
def test_cli_agg_writes_output(netcdf_file, tmp_path, name, expected_length):
    in_path, _ = netcdf_file
    out_path = tmp_path / "out.nc"
    _invoke(earthkit, name, *_io("mean", in_path, out_path))
    with xr.open_dataset(out_path) as result:
        assert "t2m" in result
        assert dict(result.sizes) == {"time": expected_length, "latitude": 2, "longitude": 3}


def test_cli_yearly_agg_values(netcdf_file, tmp_path):
    in_path, ds = netcdf_file
    out_path = tmp_path / "out.nc"
    _invoke(
        temporal_cli.yearly_agg,
        *_io("max", in_path, out_path),
        "--time-dim",
        "time",
    )
    expected = ds["t2m"].groupby("time.year").max()
    with xr.open_dataset(out_path) as result:
        np.testing.assert_allclose(result["t2m"].values, expected.values)


@pytest.mark.parametrize(
    "reduce_args",
    (
        ["--extra-reduce-dims", "latitude,longitude"],
        ["-r", "latitude", "-r", "longitude"],
    ),
)
def test_cli_extra_reduce_dims(netcdf_file, tmp_path, reduce_args):
    in_path, ds = netcdf_file
    out_path = tmp_path / "out.nc"
    _invoke(temporal_cli.yearly_agg, *_io("mean", in_path, out_path, *reduce_args))
    expected = ds["t2m"].groupby("time.year").mean(["time", "latitude", "longitude"])
    with xr.open_dataset(out_path) as result:
        assert dict(result.sizes) == {"time": 2}
        np.testing.assert_allclose(result["t2m"].values, expected.values)


def test_cli_missing_input(tmp_path):
    result = CliRunner().invoke(
        temporal_cli.yearly_agg,
        [str(a) for a in _io("mean", tmp_path / "missing.nc", tmp_path / "out.nc")],
    )
    assert result.exit_code == 2
    assert "Invalid value for 'SOURCE'" in result.output and "does not exist" in result.output


def test_cli_cds_source(netcdf_file, tmp_path, monkeypatch):
    import earthkit.data as ekd

    _, ds = netcdf_file
    calls = []

    class _Data:
        def to_xarray(self, **kwargs):
            return ds

    def _from_source(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return _Data()

    monkeypatch.setattr(ekd, "from_source", _from_source)

    request = {"variable": "2m_temperature", "year": ["2019", "2020"]}
    out_path = tmp_path / "out.nc"
    _invoke(
        earthkit,
        "yearly-agg",
        "mean",
        "cds:" + json.dumps({"dataset": "reanalysis-era5-single-levels", **request}),
        out_path,
    )
    assert calls == [("cds", ("reanalysis-era5-single-levels",), {"request": request})]
    with xr.open_dataset(out_path) as result:
        assert dict(result.sizes) == {"time": 2, "latitude": 2, "longitude": 3}
