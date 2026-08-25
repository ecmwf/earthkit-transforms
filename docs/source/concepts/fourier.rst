Fourier transforms (FFT)
------------------------

The :doc:`../autodocs/earthkit.transforms.fourier` module provides tools for computing the
discrete Fourier Transform of a data object, by means of a fast Fourier transform (FFT).

A Fourier transform decomposes a signal into the sinusoidal components that make it up,
re-expressing data sampled along a dimension (time, longitude, ...) as a spectrum indexed by
**frequency**. It is the standard tool for finding periodic structure in data — a diurnal or
seasonal cycle in a time series, or a zonal wavenumber in a field along longitude — and for
frequency-domain operations such as filtering. The inverse transform reconstructs the original
signal from its spectrum.

The array API ``fft`` extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**earthkit-transforms** exposes the full set of functions defined by the ``fft`` extension of the
`Python array API standard
<https://data-apis.org/array-api/latest/extensions/fourier_transform_functions.html>`_:

- **one-dimensional transforms**: `fft`/`ifft`, the real transforms `rfft`/`irfft`, and the
  Hermitian transforms `hfft`/`ihfft`;
- **n-dimensional transforms**: `fftn`/`ifftn` and their real counterparts `rfftn`/`irfftn`,
  for transforming over several dimensions at once (for example latitude and longitude together
  for a 2-D spatial spectrum);
- **sample-frequency helpers**: `fftfreq` and `rfftfreq`, which return the sample frequencies as
  ``xarray.DataArray`` objects;
- **spectrum shifts**: `fftshift` and `ifftshift`, which reorder a spectrum so that the
  zero-frequency component is centred.

Because the transforms call the corresponding methods of the array namespace of the input data,
the computation runs on the native backend of that data (for example NumPy or a GPU-backed array
library) and returns ``xarray.DataArray``/``xarray.Dataset`` objects. Dask-backed data stays
lazy, and the output keeps the precision of the input, so ``float32`` data gives a ``complex64``
spectrum.

The generic module operates along an explicitly named dimension, which makes it suited to
spatial or any other non-temporal analysis::

   import earthkit.transforms as ekt

   # Zonal wavenumber spectrum along longitude
   spectrum = ekt.fourier.rfft(dataarray, dim="longitude")
   restored = ekt.fourier.irfft(spectrum)

   # 2-D spectrum over latitude and longitude
   spectrum2d = ekt.fourier.rfftn(dataarray, dims=["latitude", "longitude"])
   restored2d = ekt.fourier.irfftn(spectrum2d)

For transforms along the time dimension the :doc:`temporal` module offers the same 1-D transforms,
frequency helpers and spectrum shifts as thin wrappers that detect the time dimension
automatically and attach a convenience ``period`` coordinate — see
:ref:`temporal-spectral-analysis`.

Reading the results
^^^^^^^^^^^^^^^^^^^^

A few behaviours are shared by every transform and are worth knowing before reading the results.

Applying a transform moves the transformed dimension to the **end** of the result, so a
``(time, location)`` input comes back as ``(location, frequency)``. When the input is a
``Dataset``, any variable that does not have the transformed dimension — a land-sea mask or an
orography field, say — is **passed through untransformed** rather than causing an error.

The frequency coordinate is built from the sample spacing of the transformed dimension. That
spacing is in **seconds — and the frequencies therefore in Hz — only when it is inferred from a
datetime coordinate** (``numpy.datetime64`` or ``cftime``, so non-standard calendars such as
``360_day`` are included). For a numeric coordinate, or an explicit ``sample_spacing``, the
frequencies are in the reciprocal of whatever units that coordinate or value is in, and the
coordinate is left unlabelled unless ``sample_spacing_units`` says otherwise.

Each frequency coordinate also records the dimension and the transform length it was derived
from, so the inverse transforms restore both without being told. This is what allows `irfft`,
`hfft` and `irfftn` to recover a signal of odd length, which is otherwise ambiguous: a spectrum
of length ``m`` corresponds to a real signal of either ``2m - 2`` or ``2m - 1`` points. That
record is only trusted while it remains consistent with the size of the frequency dimension, so a
spectrum that has been sliced or filtered falls back to the conventional ``2m - 2``, and ``n``
(or ``s`` for the n-dimensional transforms) can always be used to state the output length
explicitly.

.. note::

   **NaN handling.** The FFT is not NaN-aware. Because every output point depends on every input
   point, a single missing value along the transformed dimension propagates to the **whole**
   transformed axis, turning the entire spectrum — or, for an inverse transform, the entire
   reconstructed signal — into ``NaN``. Fill or interpolate gaps before transforming, for example
   with :meth:`xarray.DataArray.interpolate_na`.

   **Chunking under dask.** The transform is taken along the whole of the transformed dimension,
   so that dimension must lie within a **single dask chunk**. A dask-backed array that is chunked
   along the transformed dimension raises a ``ValueError``; rechunk it so the dimension is
   contiguous first, for example ``dataarray.chunk({"time": -1})``. The other dimensions may be
   chunked freely and are transformed in parallel.

.. _fourier-amplitudes:

Getting physical amplitudes and avoiding leakage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transforms return the raw FFT coefficients, using ``norm="backward"`` by default (no scaling
on the forward transform); ``"ortho"`` and ``"forward"`` are also accepted. To recover the
physical amplitude of a component from a real signal of length ``N``, divide the magnitude of the
corresponding `rfft` coefficient by ``N`` and double every term except the zero-frequency (and
Nyquist) term, since the negative frequencies are folded onto the positive ones. A power spectrum
is then ``|coefficient| ** 2`` after that scaling, rather than the raw ``|FFT| ** 2``.

The FFT assumes the signal is periodic over the sampled window. Real records rarely are, so a
trend or a non-integer number of cycles leaks energy across neighbouring frequencies. Removing
the mean (or a linear trend) before transforming, and applying a window function such as a Hann
window to taper the ends of the record, both reduce this leakage at the cost of some frequency
resolution.

API reference
^^^^^^^^^^^^^

The full signatures are documented in the API reference,
:doc:`../autodocs/earthkit.transforms.fourier`.

.. dropdown:: Show API documentation for ``fft``

   .. autofunction:: earthkit.transforms.fourier.fft
      :no-index:

.. dropdown:: Show API documentation for ``ifft``

   .. autofunction:: earthkit.transforms.fourier.ifft
      :no-index:

.. dropdown:: Show API documentation for ``rfft``

   .. autofunction:: earthkit.transforms.fourier.rfft
      :no-index:

.. dropdown:: Show API documentation for ``irfft``

   .. autofunction:: earthkit.transforms.fourier.irfft
      :no-index:

.. dropdown:: Show API documentation for ``hfft``

   .. autofunction:: earthkit.transforms.fourier.hfft
      :no-index:

.. dropdown:: Show API documentation for ``ihfft``

   .. autofunction:: earthkit.transforms.fourier.ihfft
      :no-index:

.. dropdown:: Show API documentation for ``fftn``

   .. autofunction:: earthkit.transforms.fourier.fftn
      :no-index:

.. dropdown:: Show API documentation for ``ifftn``

   .. autofunction:: earthkit.transforms.fourier.ifftn
      :no-index:

.. dropdown:: Show API documentation for ``rfftn``

   .. autofunction:: earthkit.transforms.fourier.rfftn
      :no-index:

.. dropdown:: Show API documentation for ``irfftn``

   .. autofunction:: earthkit.transforms.fourier.irfftn
      :no-index:

.. dropdown:: Show API documentation for ``fftfreq``

   .. autofunction:: earthkit.transforms.fourier.fftfreq
      :no-index:

.. dropdown:: Show API documentation for ``rfftfreq``

   .. autofunction:: earthkit.transforms.fourier.rfftfreq
      :no-index:

.. dropdown:: Show API documentation for ``fftshift``

   .. autofunction:: earthkit.transforms.fourier.fftshift
      :no-index:

.. dropdown:: Show API documentation for ``ifftshift``

   .. autofunction:: earthkit.transforms.fourier.ifftshift
      :no-index:
