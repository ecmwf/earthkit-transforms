Welcome to Earthkit-transforms’s documentation!
===============================================

**earthkit-transforms** is a library of software tools to support people
working with climate and meteorology data. It is made of a sub-package
called **earthkit-transforms.aggregate** which includes methods for aggregating
data in time and space. **earthkit-transforms** has been designed
following the philosophy of earthkit, hence the methods should be
interoperable with any data object understood by earthkit-data.

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

    from earthkit.transforms import aggregate

    daily_mean = aggregate.temporal.daily_mean(MY_DATA)


.. toctree::
   :caption: Examples
   :maxdepth: 2

   aggregate-examples

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
