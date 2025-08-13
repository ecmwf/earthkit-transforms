from earthkit import transforms


def test_version() -> None:
    assert transforms.__version__ != "999"
