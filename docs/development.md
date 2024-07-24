# Development

## Contributions

The code repository is hosted on `Github`\_, testing, bug reports and contributions are highly welcomed and appreciated. Feel free to fork it and submit your PRs against the **develop** branch.

### Development setup

The recommended development environment is based on **conda**.

First, clone the repository locally. You can use the following command:

.. code-block:: shell

git clone --branch develop git@github.com:ecmwf/earthkit-transforms.git

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.11:

```
conda create -n DEVELOP -c conda-forge python=3.11
conda activate DEVELOP
make conda-env-update
```

Before pushing to GitHub, run the following commands:

1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make default`

### Run the quality assurance checks independently

If you are interested on one specific aspect of the quality assurance tests they can be run independantly as:

1. Coding standards test `make qa`
1. Unit tests `make unit-test`
1. Documentation build `make docs-build`
