Dimension detection
===================

**earthkit.transforms** makes computations in given domains easier by detecting the dimensions and
coordinates of the input data in which it operates. The detection is based on CF conventions, and some
additional common identifiers. Users are encouraged to ensure their data is CF compliant to get the
best experience.

The dimension/coordinate detection is performed as a first match of the following checks:

1. `axis` attribute on the data variable or coordinate, e.g. `axis = 'T'` for time. `This is a CF convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#coordinate-types>`_.
2. `standard_name` attribute on the coordinate, e.g. `standard_name = 'time'` for time. `This is a CF convention <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#coordinate-types.`_.
3. Common variable names for coordinates, e.g. `time`, `latitude`, `longitude`, `lat`, `lon`

The following table summarises the identifiers used for dimension/coordinate detection:

.. list-table::
  :width: 100 %
  :widths: 25 25 25 25 25
  :header-rows: 1

  * - Identifier
    - X (Longitude)
    - Y (Latitude)
    - T (Time)
    - Ensemble

  * - ``axis`` attribute
    - "X"
    - "Y"
    - "T"
    - "E"

  * - ``standard_name`` attribute
    - "projection_x_coordinate", "longitude", "grid_longitude"
    - "projection_y_coordinate", "latitude", "grid_latitude"
    - "time", "valid_time", "forecast_reference_time"
    - "realization"

  * - Common variable names
    - "lon", "long", "longitude"
    - "lat", "latitude"
    - "time", "valid_time", "forecast_reference_time"
    - "ensemble_member", "ensemble", "member", "number", "realization", "realisation"