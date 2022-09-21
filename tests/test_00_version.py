import tyto


def test_version() -> None:
    assert tyto.__version__ != "999"
