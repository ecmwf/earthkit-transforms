Welcome to Earthkit-transforms’s documentation!
===============================================

**earthkit-transforms** is a library of software tools to support people
working with climate and meteorology data. It is made of sub-modules desigined
for  transformations of data in specific domains.
For example, the temporal module contains methods for aggregation and
statistics analysis accross time dimensions/coordinates.
**earthkit-transforms** has been designed
following the philosophy of earthkit, hence the methods should be
interoperable with any data object understood by earthkit-data.
Under the bonnet, **earthkit-transforms** generally uses
[xarray](https://xarray.pydata.org/en/stable/) for computations,
therefore it recommended to prepare your data as xarray objects
before using the methods. This ensure clarity and consistency of the
output produced by **earthkit-transforms**.

Getting started
---------------

**earthkit-transforms** is available from PyPi. To make use of the
interoperable functionality you should ensure that you have installed
the dependencies of *earthkit-data*.

Installation
~~~~~~~~~~~~

.. code:: bash

   pip install earthkit-transforms

Import and use
~~~~~~~~~~~~~~

.. code:: python

    import earthkit.transforms as ekt

    daily_mean = ekt.temporal.daily_mean(MY_DATA)


.. toctree::
   :caption: Examples
   :maxdepth: 2

   notebooks/aggregate-examples

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   guide/index
   _api/index
   development

.. toctree::
   :maxdepth: 2
   :caption: Installation

   install
   release-notes/index
   licence


.. toctree::
   :maxdepth: 2
   :caption: Projects

   earthkit <https://earthkit.readthedocs.io/en/latest>


Indices and tables
==================

* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
