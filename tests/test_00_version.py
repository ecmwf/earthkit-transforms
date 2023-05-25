from earthkit import climate


def test_version() -> None:
    assert climate.__version__ != "999"
