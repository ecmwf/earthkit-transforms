from earthkit import data as ek_data
from earthkit.aggregate import shapes
from earthkit.data.testing import earthkit_remote_test_data_file

REMOTE_ERA5_FILE = earthkit_remote_test_data_file("test-data", "era5_temperature_europe_2015.grib")
ERA5_DATA = ek_data.from_source("url", REMOTE_ERA5_FILE)
DATASET = ERA5_DATA.to_xarray()
DATAARRAY = DATASET.t2m

REMOTE_NUTS_FILE = earthkit_remote_test_data_file("test-data", "NUTS_RG_60M_2021_4326_LEVL_0.geojson")
NUTS_DATA = ek_data.from_source("url", REMOTE_NUTS_FILE)
GEODATAFRAME = NUTS_DATA.to_pandas()


def test_shapes_masks():
    # With a dataarray
    masked_data = shapes.masks(DATAARRAY, GEODATAFRAME)
    assert isinstance(masked_data, type(DATAARRAY))
    assert "FID" in masked_data.dims
    assert len(masked_data["FID"]) == len(GEODATAFRAME)

    # With a dataset
    # masked_data = shapes.masks(DATASET, GEODATAFRAME)
