# Copyright (c) AIoWay Authors - All Rights Reserved

"A module for composite `Dset`s."

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import TSpecCompat

from .dsets import Dset, IdxDset

__all__ = [
    "CompositeDset",
    "ComposedDset",
    "IdxComposedDset",
    "InputTarget",
    "InputTargetLikeDset",
    "InputTargetDset",
]


class CompositeDset(Dset, abc.ABC):
    """
    A `Dset` that the data should be composites.
    """

    if typing.TYPE_CHECKING:

        @typing.override
        @abc.abstractmethod
        def __tspec__(self) -> TSpecCompat:
            raise NotImplementedError


class ComposedDset(CompositeDset):
    def __init__(self, **dsets: Dset) -> None:
        self._dsets = dsets

    @typing.override
    def __tspec__(self) -> tspecs.Composite:
        "The composed spec would be a composite."

        return tspecs.Composite(
            {key: val.__tspec__() for key, val in self.dsets.items()}
        )

    @property
    def dsets(self) -> cabc.Mapping[str, Dset]:
        return self._dsets


class IdxComposedDset(ComposedDset, IdxDset):
    if typing.TYPE_CHECKING:

        @property
        def dsets(self) -> cabc.Mapping[str, IdxDset]: ...

    def __init__(self, **dsets: IdxDset) -> None:

        super().__init__(**dsets)

        if (lengths := len({len(dset) for dset in dsets.values()})) != 1:
            raise ValueError(
                f"The lengths of the given datasets do not match. Found {lengths=}."
            )

    def __len__(self) -> int:
        return len(next(iter(self.dsets.values())))

    def __getitem__(self, idx):
        return td.TensorDict({k: d[idx] for k, d in self.dsets.items()})

    def __getitems__(self, idx):
        return td.TensorDict({k: d.__getitems__(idx) for k, d in self.dsets.items()})


class InputTarget(td.TensorClass):
    input: td.TensorDictBase | torch.Tensor
    "The input field."

    target: td.TensorDictBase | torch.Tensor
    "The target field."


class InputTargetLikeDset(IdxDset, abc.ABC):
    @property
    @abc.abstractmethod
    def input_space(self) -> tspecs.TensorSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def target_space(self) -> tspecs.TensorSpec:
        raise NotImplementedError

    def __tspec__(self) -> tspecs.Composite:
        return tspecs.Composite(input=self.input_space, target=self.target_space)


class InputTargetDset(InputTargetLikeDset):
    def __init__(self, input: IdxDset, target: IdxDset):
        self._input = input
        self._target = target

    def __getitem__(self, idx):
        return InputTarget(input=self.input[idx], target=self.target[idx])

    def __getitems__(self, idx):
        return InputTarget(
            input=self.input.__getitems__(idx), target=self.target.__getitems__(idx)
        )

    @property
    def input(self) -> IdxDset:
        return self._input

    @property
    def target(self) -> IdxDset:
        return self._target

    @property
    def input_space(self) -> tspecs.TensorSpec:
        return self.input.__tspec__()

    @property
    def target_space(self) -> tspecs.TensorSpec:
        return self.target.__tspec__()
