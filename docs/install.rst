Getting started
===============

Installing from PyPi
--------------------

It is recommended that you use the latest version of python3 and pip for
the installation of earthkit-transforms.

.. code:: bash

   pip install earthkit-transforms

To make use of the interoperable functionality you should ensure that
you have installed the *earthkit-data* dependency.


Import and use
--------------

.. code:: python

    import earthkit.transforms as ekt

    daily_mean = ekt.temporal.daily_mean(MY_DATA)

