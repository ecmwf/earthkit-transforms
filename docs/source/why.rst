Why earthkit-transforms?
==========================

**earthkit-transforms** has been designed
following the philosophy of earthkit, hence the methods should be
interoperable with any data object understood by earthkit-data.
Under the bonnet, **earthkit-transforms** generally uses
`xarray <https://xarray.pydata.org/en/stable/>`_ for computations,
therefore it recommended to prepare your data as xarray objects
before using the methods. This ensure clarity and consistency of the
output produced by **earthkit-transforms**.