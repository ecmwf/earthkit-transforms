from earthkit import aggregate


def test_version() -> None:
    assert aggregate.__version__ != "999"
