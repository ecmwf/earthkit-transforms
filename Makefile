PROJECT := earthkit-transforms
CONDA := conda
CONDAFLAGS :=
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
