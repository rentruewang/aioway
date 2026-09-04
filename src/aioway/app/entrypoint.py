# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections import abc as cabc

from torch import optim

from aioway.emits import emit, route_loss
from aioway.dsets import Dset, route_dset
from aioway.instrs import Instr
from aioway.trainers import StaticTrainer, TrainCfg
from aioway.tspecs import TSpec, as_tspec

from .server import serve

__all__ = ["add_input", "add_output", "add_module", "add_trainer"]

_INPUT_DATASETS: dict[str, Dset] = {}
"The input datasets."

_OUTPUT_SPECS: dict[str, TSpec] = {}
"The output shapes."


def add_input(name: str, dset: Dset) -> None:
    """
    Add an input source.
    """

    if name in _INPUT_DATASETS:
        raise KeyError(f"{name=} is already used by another input dataset.")

    _INPUT_DATASETS[name] = dset


@serve("add_input")
def add_input_by_name(name: str, path: str) -> None:
    dset = route_dset(path)
    add_input(name, dset)


def add_output(name: str, spec: TSpec) -> None:
    """
    Add an output sink.
    """

    if name in _OUTPUT_SPECS:
        raise KeyError(f"{name=} is already used by another output sink.")

    _OUTPUT_SPECS[name] = spec


def add_module(input: str, output: str) -> cabc.Generator[Instr]:
    input_dataset = _INPUT_DATASETS[input]
    output_spec = _OUTPUT_SPECS[output]

    yield from emit(as_tspec(input_dataset.__tspec__()), output_spec)


def add_trainer(
    input: str, output: str, label: str, cfg: TrainCfg
) -> cabc.Generator[StaticTrainer]:
    input_dataset = _INPUT_DATASETS[input]
    output_spec = _OUTPUT_SPECS[output]
    label_dataset = _INPUT_DATASETS[label]

    if operator.length_hint(input_dataset) != operator.length_hint(label_dataset):
        raise ValueError(
            f"Datasets {input=} and {label=} can not be used in supervised training "
            "because they have different lengths."
        )

    for instr in add_module(input, output):
        module = instr.module()
        yield StaticTrainer(
            cfg=cfg,
            module=module,
            optimizer=optim.AdamW(module.parameters()),
            loss_func=next(route_loss(output_spec, label_dataset.__tspec__())),
        )
