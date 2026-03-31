PROJECT := earthkit-transforms
COV_REPORT := html
NBSPHINX_EXECUTE := auto

default: qa unit-tests
qa:
	pre-commit run --all-files

unit-tests:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --ignore=tests/legacy-api

legacy-api-unit-tests:
	python -m pytest -vv \
		tests/legacy-api \
		--cov=. \
		--cov-report=$(COV_REPORT)

type-check:
	python -m mypy . --no-namespace-packages

docs-build:
	cd docs && make clean && make html SPHINXOPTS="-D nbsphinx_execute=$(NBSPHINX_EXECUTE)"

clean-pip-env:
	rm -rf .venv
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel pre-commit mypy
	.venv/bin/pip install -e .
	echo to activate the environment, run: . .venv/bin/activate

clean-uv-env:
	rm -rf .venv
	uv venv --python 3.12 .venv
	uv pip install -e .
	echo to activate the environment, run: . .venv/bin/activate

clean-conda-env:
	conda create -y -p ./.conda -c conda-forge python=3.12
	conda run -p ./.conda pip install -e .
	echo to activate the environment, run: conda activate ./.conda
