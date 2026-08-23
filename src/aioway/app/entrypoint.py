# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.dsets import Dset
from aioway.tspecs import TSpec

__all__ = ["add_source", "add_sink"]

_INPUT_DATASETS: dict[str, Dset] = {}
"The input datasets."

_OUTPUT_SPECS: dict[str, TSpec] = {}
"The output shapes."


def add_source(name: str, dset: Dset) -> None:
    """
    Add an input source.
    """

    if _already_has_name(name):
        raise KeyError(f"{name=} is already used.")

    _INPUT_DATASETS[name] = dset


def add_sink(name: str, spec: TSpec) -> None:
    """
    Add an output sink.
    """

    if _already_has_name(name):
        raise KeyError(f"{name=} is already used")

    _OUTPUT_SPECS[name] = spec


def _already_has_name(name: str):
    return name in _INPUT_DATASETS or name in _OUTPUT_SPECS
