Installation and Getting Started
================================

Installing from PyPi
--------------------

It is recommended that you use the latest version of python3 and pip for
the installation of earthkit-transforms.

For a basic installation of earthkit-transforms:

.. code:: bash

   pip install earthkit-transforms

To make use of all the functionality of earthkit-transforms,
it is recommended to install the optional dependencies:

.. code:: bash

   pip install earthkit-transforms[all]


Import and use
--------------

.. code:: python

    import earthkit.transforms as ekt

    daily_mean = ekt.temporal.daily_mean(MY_DATA)
