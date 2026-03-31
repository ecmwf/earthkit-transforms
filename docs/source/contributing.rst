Contributing
============

The code repository is hosted on ``Github``, testing, bug reports
and contributions are highly welcomed and appreciated. Feel free to fork
it and submit your PRs against the **main** branch.

Development setup
~~~~~~~~~~~~~~~~~

First, clone the repository locally. You can use the following command:


.. code-block:: shell

   git clone git@github.com:ecmwf/earthkit-tranforms.git


The Makefile provides some convenient commands for setting up clean environments for developing
and testing the package. To set up a clean `pip` environment, run the following:

::

   make clean-pip-env

   # Then activate the virtual environment with
   . venv/bin/activate

Or if you prefer to use `uv`, then use the following:

::

   make clean-uv-env

   # Then activate the virtual environment with
   uv venv .venv

Or if you prefer to use `conda`, then use the following:

::

   make clean-conda-env

   # Then activate the virtual environment with
   conda activate ./.conda


The environment created has the base installation of the package.
To replicate the environment used in the CI, install the package with the `ci` extra:

::

   pip install -e ".[ci]"


Run the CI checks independently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the `ci` extra installed, you can run the CI checks locally with:

::

   make default

If you are interested on one specific aspect of the quality assurance
tests they can be run independently as:

1. Coding standards test ``make qa``
2. Unit tests ``make unit-test``
3. Documentation build ``make docs-build``
4. Type checking ``make type-check``
