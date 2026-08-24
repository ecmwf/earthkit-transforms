Temporal computations
---------------------

The :doc:`../autodocs/earthkit.transforms.temporal` module includes methods
for transforming data with respect to the temporal coordinate(s).
This includes aggregating the data in time dimension to a single value,
daily values, or monthly values and calculating rates from accumulated data.

Specifying the time dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every temporal method operates along a single time dimension (or coordinate). By default
this dimension is detected automatically from the metadata of the data object, so in most
cases you do not need to provide it explicitly. The detection follows this order:

1. A dimension whose ``axis`` attribute is ``"T"``.
2. A dimension whose ``standard_name`` attribute is a CF time name, i.e. one of
   ``"time"``, ``"valid_time"`` or ``"forecast_reference_time"``.
3. A dimension whose name matches one of the recognised time names ``"time"``,
   ``"valid_time"`` or ``"forecast_reference_time"``.

If none of these can be found, or if the automatically detected dimension is not the one
you want to aggregate over, you can override the detection using the ``time_dim`` parameter.
This is common when your data uses a non-standard name for the time coordinate (for example
``"forecast_time"`` or ``"step"``), or when the object contains more than one time-like
coordinate and you need to select a specific one::

   # Aggregate along a coordinate named "forecast_time"
   earthkit.transforms.temporal.daily_mean(dataarray, time_dim="forecast_time")

The ``time_dim`` parameter is accepted by every function in the temporal module, including
``reduce``, the ``mean``/``sum``/``min``/``max``/``std`` aggregations, the ``daily_*`` and
``monthly_*`` methods, and ``accumulation_to_rate``. A worked example is available in the
how-to guide :doc:`../how-tos/temporal/howto_specify_time_dim`.

Aggregation methods
^^^^^^^^^^^^^^^^^

To aggregate the data in time dimension to a single value you can use the `temporal.reduce`,
`temporal.mean`, `temporal.sum`, `temporal.min`, `temporal.max` functions. These functions
take an xarray data object and return the aggregated value of the data. The time dimension
is automatically detected based on the metadata of the data object, to override this you can
use the `time_dim` parameter (see `Specifying the time dimension`_ above).

.. dropdown:: Show API documentation for ``reduce``

   .. autofunction:: earthkit.transforms.temporal.reduce
      :no-index:


**Daily and monthly aggregations**

To calculate calculate daily/monthly values you can use the `temporal.daily_reduce` or
`temporal.monthly_reduce` functions. These functions reduce the data in time dimension to
daily or monthly values respectively. The `how` parameter can be used to specify the aggregation
method. The default is `mean`.

.. dropdown:: Show API documentation for ``daily_reduce``

   .. autofunction:: earthkit.transforms.temporal.daily_reduce
      :no-index:

.. dropdown:: Show API documentation for ``monthly_reduce``

   .. autofunction:: earthkit.transforms.temporal.monthly_reduce
      :no-index:


In addition to the `XXX_reduce` functions, the temporal module also includes several methods
which calculate the desired reduction, without the "how" parameter. These methods are
wrappers of the `daily_reduce` and `monthly_reduce` methods and are documented
in the API reference guide: :doc:`../autodocs/earthkit.transforms.temporal`:

- `temporal.daily_mean`
- `temporal.daily_median`
- `temporal.daily_min`
- `temporal.daily_max`
- `temporal.daily_sum`
- `temporal.daily_std`
- `temporal.monthly_mean`
- `temporal.monthly_median`
- `temporal.monthly_min`
- `temporal.monthly_max`
- `temporal.monthly_sum`
- `temporal.monthly_std`


Rate calculations
^^^^^^^^^^^^^^^^^

To calculate rates from accumulated data you can use the `temporal.accumulation_to_rate` function.
This function takes an xarray data object and returns the rate of change of the data. The time dimension
is automatically detected based on the metadata of the data object, to override this
you can use the `time_dim` parameter. Similarly, the step between time points is automatically
detected, but can be overridden using the `step` parameter. If using the step parameter, the value
should be provided in hours unless specifying a different unit with the `step_units` parameter.
`step_units` can take any time units recognised by pandas, e.g. "minutes", "days", "15min", "3H", etc.

By default the function will calculate the rate per second, but this can be changed using the
`rate_units` parameter. `rate_units` can take any time units recognised by pandas,
e.g. "minutes", "days", "15min", "3H", etc. If you are only interested in "deaccumulating" the data,
i.e. converting accumulated values to step values, you can set `rate_units` to "step_length" such that the
`rate_units` will be equal to the step length between time points.

The `accumulation_type` parameter is used to specify the type of accumulation used in the input data.

The options are:

- "start_of_step": accumulation restarts at the beginning of each time step, e.g. ERA5.
- "start_of_forecast": accumulation restarts at the beginning of each forecast, e.g. Seeason forecasts.
- "start_of_day": accumulation restarts at the beginning of each day, e.g. ERA5-land.

.. dropdown:: Show API documentation for ``accumulation_to_rate``

   .. autofunction:: earthkit.transforms.temporal.accumulation_to_rate
      :no-index:


Spectral analysis (FFT)
^^^^^^^^^^^^^^^^^^^^^^^^

The temporal module provides a full set of entry points for computing the discrete Fourier
Transform (FFT) of a data object along its time dimension. These are ``xarray`` wrappers for
the functions in the ``fft`` extension of the Python array API standard. As with the other
temporal methods, the time dimension is automatically detected from the metadata of the data
object and can be overridden with the `time_dim` parameter.

The transforms are implemented using the Python array API standard, applying the corresponding
methods of the array namespace of the input data. This means the computation runs on the native
backend of the data (for example NumPy or a GPU-backed array library) and returns
``xarray.DataArray``/``xarray.Dataset`` objects. Dask-backed data stays lazy, and the output
keeps the precision of the input, so ``float32`` data gives a ``complex64`` spectrum.

Two things are worth knowing before reading the results. Applying a transform moves the
transformed dimension to the **end** of the result, so a ``(time, location)`` input comes back
as ``(location, frequency)``. And when the input is a ``Dataset``, any variable that does not
have the time dimension — a land-sea mask or an orography field, say — is **passed through
untransformed** rather than causing an error.

**Complex transforms**

Use `temporal.fft` for the forward transform and `temporal.ifft` for the inverse transform.
For the forward transform, the time dimension is replaced by a ``frequency`` dimension, and the
result is complex-valued. The inverse transform, `temporal.ifft`, operates on the ``frequency``
dimension and returns the data to the time domain::

   spectrum = earthkit.transforms.temporal.fft(dataarray)
   restored = earthkit.transforms.temporal.ifft(spectrum, time_coord=dataarray["time"])

The forward transforms record the dimension and the transform length they were derived from on
the ``frequency`` coordinate, so the inverse transforms restore both without being told. A round
trip therefore returns the dimension the data started with, whether it was called ``time``,
``valid_time`` or anything else.

.. note::

   The frequency coordinate is derived from the spacing of the time coordinate. That spacing is
   in **seconds — and the frequencies therefore in Hz — only when the time coordinate is a
   datetime** (``numpy.datetime64`` or ``cftime``, so non-standard calendars such as ``360_day``
   are included). For a numeric time coordinate, such as one holding *hours since* some epoch,
   the frequencies are in the reciprocal of that coordinate's own units, and the ``frequency``
   and ``period`` coordinates are left unlabelled rather than claiming units that were never
   established. The same applies when `sample_spacing` is given explicitly; pass
   `sample_spacing_units` alongside it to label the result.

**Real and Hermitian transforms**

For real-valued input, `temporal.rfft` returns only the non-negative frequency terms and
`temporal.irfft` performs the corresponding inverse. The Hermitian transforms `temporal.hfft`
and `temporal.ihfft` are provided for signals with Hermitian symmetry.

A spectrum of length ``m`` corresponds to a real signal of either ``2m - 2`` or ``2m - 1``
points, so inverting one is ambiguous in general. Because the forward transforms record the
length they were built from, `temporal.irfft` and `temporal.hfft` recover the original number
of time steps exactly, for odd and even lengths alike. That record is only trusted while it
remains consistent with the size of the frequency dimension: a spectrum that has been sliced or
filtered falls back to the conventional ``2m - 2``, and `n` can always be used to state the
output length explicitly.

**N-dimensional transforms**

The n-dimensional transforms `temporal.fftn`, `temporal.ifftn`, `temporal.rfftn` and
`temporal.irfftn` are also available. When the transform dimensions are not given explicitly, the
forward transforms default to operating over the detected time dimension. Passing ``dims``
explicitly transforms over several axes at once, which is how a two-dimensional space-time or
spatial spectrum is computed::

   # 2-D spectrum over latitude and longitude
   spectrum = earthkit.transforms.temporal.rfftn(dataarray, dims=["latitude", "longitude"])
   restored = earthkit.transforms.temporal.irfftn(spectrum)

**Frequency helpers and spectrum shifts**

The sample-frequency helpers `temporal.fftfreq` and `temporal.rfftfreq` return the sample
frequencies as ``xarray.DataArray`` objects, and the spectrum shifts `temporal.fftshift` and
`temporal.ifftshift` reorder a spectrum so that the zero-frequency component is centred. The
helpers take a bare `sample_spacing` with no coordinate to infer units from, so pass
`sample_spacing_units="s"` alongside a spacing in seconds if the result should be labelled in Hz.

A generic set of entry points that operate along a user-specified dimension is also available in
the `earthkit.transforms.fourier` module, for example to compute the FFT along a spatial
dimension. It exposes the same set of functions (`fft`/`ifft`, `rfft`/`irfft`, `hfft`/`ihfft`,
`fftn`/`ifftn`/`rfftn`/`irfftn`, `fftfreq`/`rfftfreq` and `fftshift`/`ifftshift`) without the
automatic time-dimension detection. The dimension is named explicitly instead::

   # Zonal wavenumber spectrum along longitude
   spectrum = earthkit.transforms.fourier.rfft(dataarray, dim="longitude")
   restored = earthkit.transforms.fourier.irfft(spectrum)

**Getting physical amplitudes and avoiding leakage**

The transforms return the raw FFT coefficients, using ``norm="backward"`` by default (no scaling
on the forward transform); ``"ortho"`` and ``"forward"`` are also accepted. To recover the
physical amplitude of a component from a real signal of length ``N``, divide the magnitude of
the corresponding `rfft` coefficient by ``N`` and double every term except the zero-frequency
(and Nyquist) term, since the negative frequencies are folded onto the positive ones. A power
spectrum is then ``|coefficient| ** 2`` after that scaling, rather than the raw ``|FFT| ** 2``.

The FFT assumes the signal is periodic over the sampled window. Real records rarely are, so a
trend or a non-integer number of cycles leaks energy across neighbouring frequencies. Removing
the mean (or a linear trend) before transforming, and applying a window function such as a Hann
window to taper the ends of the record, both reduce this leakage at the cost of some frequency
resolution.

.. dropdown:: Show API documentation for ``fft``

   .. autofunction:: earthkit.transforms.temporal.fft
      :no-index:

.. dropdown:: Show API documentation for ``ifft``

   .. autofunction:: earthkit.transforms.temporal.ifft
      :no-index:

.. dropdown:: Show API documentation for ``rfft``

   .. autofunction:: earthkit.transforms.temporal.rfft
      :no-index:

.. dropdown:: Show API documentation for ``irfft``

   .. autofunction:: earthkit.transforms.temporal.irfft
      :no-index:

.. dropdown:: Show API documentation for ``hfft``

   .. autofunction:: earthkit.transforms.temporal.hfft
      :no-index:

.. dropdown:: Show API documentation for ``ihfft``

   .. autofunction:: earthkit.transforms.temporal.ihfft
      :no-index:

.. dropdown:: Show API documentation for ``fftn``

   .. autofunction:: earthkit.transforms.temporal.fftn
      :no-index:

.. dropdown:: Show API documentation for ``ifftn``

   .. autofunction:: earthkit.transforms.temporal.ifftn
      :no-index:

.. dropdown:: Show API documentation for ``rfftn``

   .. autofunction:: earthkit.transforms.temporal.rfftn
      :no-index:

.. dropdown:: Show API documentation for ``irfftn``

   .. autofunction:: earthkit.transforms.temporal.irfftn
      :no-index:

.. dropdown:: Show API documentation for ``fftfreq``

   .. autofunction:: earthkit.transforms.temporal.fftfreq
      :no-index:

.. dropdown:: Show API documentation for ``rfftfreq``

   .. autofunction:: earthkit.transforms.temporal.rfftfreq
      :no-index:

.. dropdown:: Show API documentation for ``fftshift``

   .. autofunction:: earthkit.transforms.temporal.fftshift
      :no-index:

.. dropdown:: Show API documentation for ``ifftshift``

   .. autofunction:: earthkit.transforms.temporal.ifftshift
      :no-index:
