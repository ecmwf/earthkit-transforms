earthkit.transforms.spatial
--------------------------------------

The :doc:`../../_api/transforms/spatial/index` module includes methods
for aggregating data in space. This includes masking and aggregating data with geometries.

To mask data you can use the :func:`mask` function. This function takes an xarray
data object and a geometry object and returns the data object with the values
outside the geometry masked.

.. autofunction:: earthkit.transforms.spatial.mask


To calculate an aggregated value for the data within a geometry you can use the
:func:`reduce` function. This function takes an xarray data object and a geometry
object and returns the aggregated value of the data within the geometry. The `how`
parameter can be used to specify the aggregation method. The default is `mean`.

.. autofunction:: earthkit.transforms.spatial.reduce


In addition to the above functions, the spatial module also includes several methods
for computing the intermedieate steps of the aggregation process. These methods are
documented in the API reference guide: :doc:`../../_api/transforms/spatial/index`
