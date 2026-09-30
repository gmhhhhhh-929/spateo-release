"""Native reader registry; implementations live in individual platform modules."""

from importlib import import_module

DOMESTIC = frozenset({"seekspace", "bmkmanu", "salus", "singleron"})


def get_reader(technology):
    if technology not in DOMESTIC:
        raise ValueError(f"No domestic native reader for {technology!r}")
    return import_module(f"{__package__}._{technology}")


def discover_domestic(files, requested):
    return [candidate for tech in sorted(DOMESTIC) for candidate in get_reader(tech).discover(files, requested)]
