Welcome to Earthkit-transforms' documentation!
==============================================

**earthkit-transforms** is a library of software tools to support people
working with climate and meteorology data. It is made of sub-modules designed
for  transformations of data in specific domains.
For example, the temporal module contains methods for aggregation and
statistics analysis accross time dimensions/coordinates.

.. **earthkit-transforms** has been designed
.. following the philosophy of earthkit, hence the methods should be
.. interoperable with any data object understood by earthkit-data.
.. Under the bonnet, **earthkit-transforms** generally uses
.. `xarray <https://xarray.pydata.org/en/stable/>`_ for computations,
.. therefore it recommended to prepare your data as xarray objects
.. before using the methods. This ensure clarity and consistency of the
.. output produced by **earthkit-transforms**.

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: Installation and Getting Started
      :img-top: _static/rocket.svg
      :link: getting-started
      :link-type: doc
      :class-card: sd-shadow-sm

      New to earthkit-transforms? Start here with installation and a quick overview.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Tutorials
      :img-top: _static/book.svg
      :link: tutorials/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Step-by-step guides to learn earthkit-transforms.

   .. grid-item-card:: How-tos
      :img-top: _static/tool.svg
      :link: how-tos
      :link-type: doc
      :class-card: sd-shadow-sm

      Practical recipes for common tasks.

   .. grid-item-card:: Concepts and Explanations
      :img-top: _static/bulb.svg
      :link: explanations/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand the core ideas behind earthkit-transforms.

   .. grid-item-card:: API Reference Guide
      :img-top: _static/brackets-contain.svg
      :link: api-reference
      :link-type: doc
      :class-card: sd-shadow-sm

      Detailed documentation of all functions and classes.


.. toctree::
   :caption: User guide
   :maxdepth: 2
   :hidden:

   getting-started
   tutorials/index
   how-tos
   explanations/index
   api-reference


.. toctree::
   :caption: Developer guide
   :maxdepth: 2
   :hidden:

   contributing


.. toctree::
   :maxdepth: 2
   :caption: Extras
   :hidden:

   release-notes/index
   licence
   genindex


.. toctree::
   :maxdepth: 2
   :caption: Related projects
   :hidden:

   earthkit <https://earthkit.readthedocs.io/en/latest>
   earthkit-data <https://earthkit-data.readthedocs.io/en/latest>
   earthkit-plots <https://earthkit-plots.readthedocs.io/en/latest>
   earthkit-meteo <https://earthkit-meteo.readthedocs.io/en/latest>
   earthkit-hydro <https://earthkit-hydro.readthedocs.io/en/latest>
