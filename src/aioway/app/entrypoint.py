# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.dsets import Dset
from aioway.tspecs import TSpec

__all__ = ["add_input", "add_output", "add_label"]

_INPUT_DATASETS: dict[str, Dset] = {}
"The input datasets."

_OUTPUT_SPECS: dict[str, TSpec] = {}
"The output shapes."

_LABEL_DATASETS: dict[str, Dset] = {}
"The label dataset."


def add_input(name: str, dset: Dset) -> None:
    """
    Add an input source.
    """

    if name in _INPUT_DATASETS:
        raise KeyError(f"{name=} is already used by another input dataset.")

    _INPUT_DATASETS[name] = dset


def add_output(name: str, spec: TSpec) -> None:
    """
    Add an output sink.
    """

    if name in _OUTPUT_SPECS:
        raise KeyError(f"{name=} is already used by another output sink.")

    _OUTPUT_SPECS[name] = spec


def add_label(name: str, dset: Dset) -> None:
    """
    Add a training label dataset.
    """

    if name in _LABEL_DATASETS:
        raise KeyError(f"{name=} is already used by another label dataset.")

    _LABEL_DATASETS[name] = dset
