Version 0.4 Updates
/////////////////////////

Version 0.4.0
=============

**Breaking change:**

`earthkit.transforms[.aggregate].spatial.mask` API changed in this version (in '38).
The methods `mask` and `masks` were merged into a single method `mask` that covers the functionality of both.
The default behaviour of `mask` is to return a mask for each feature found in the geometry object,
i.e. the same behaviour as the `masks` method in earlier versions. To reproduce the earlier behaviour
of `mask`, i.e. to union all features into a single mask, then use the new kwarg `union_geometries=True`.
For more details, please see the
:doc:`spatial aggregation example notebook <../notebooks/spatial/01-era5-masking>`.

- Update earthkit-transforms to use latest version of earthkit-data (in #37)
- Precomputed masks for spatial aggregation methods (in #38).
- Refactor and updates of repository structure and linting/typesetting (in #41, #42, #44, #45)
- Split reduce dataarray into separate functions for better handling of pandas and xarray outputs (in #46)
