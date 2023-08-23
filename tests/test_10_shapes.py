import xarray as xr
import numpy as np
import geopandas as gpd
from earthkit.climate import shapes

DATASET = xr.open_dataset(
    "tests/data/era5_temperature_europe_2015.grib", chunks={'time': 48}
)
DATAARRAY = DATASET.t2m
GEODATAFRAME = gpd.read_file(
    "tests/data/nuts/NUTS_RG_60M_2021_4326_LEVL_0.geojson"
)

def test_shapes_masks():
    # With a dataarray
    masked_data = shapes.masks(DATAARRAY, GEODATAFRAME)
    assert isinstance(masked_data, type(DATAARRAY))
    assert "FID" in masked_data.dims
    assert len(masked_data["FID"])==len(GEODATAFRAME)

    # With a dataset
    # masked_data = shapes.masks(DATASET, GEODATAFRAME)


