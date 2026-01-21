from earthkit import transforms as ekt


def test_aggregate_submodule_imported() -> None:
    dir(ekt.aggregate)
    dir(ekt.aggregate.climatology)
    dir(ekt.aggregate.spatial)
    dir(ekt.aggregate.temporal)
