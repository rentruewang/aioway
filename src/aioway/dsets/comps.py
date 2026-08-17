# Copyright (c) AIoWay Authors - All Rights Reserved

"A module for composite `Dset`s."

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torchrl import data as rldata

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
        def __spec__(self) -> rldata.Composite:
            raise NotImplementedError


class ComposedDset(CompositeDset):
    def __init__(self, **dsets: Dset) -> None:
        self._dsets = dsets

    @typing.override
    def __spec__(self) -> rldata.Composite:
        "The composed spec would be a composite."

        return rldata.Composite(
            {key: val.__spec__() for key, val in self.dsets.items()}
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
    def input_spec(self) -> rldata.TensorSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def target_spec(self) -> rldata.TensorSpec:
        raise NotImplementedError

    def __spec__(self) -> rldata.Composite:
        return rldata.Composite(input=self.input_spec, target=self.target_spec)


class InputTargetDset(InputTargetLikeDset):
    def __init__(self, input: IdxDset, target: IdxDset):
        super().__init__(input=input, target=target)

    def __getitem__(self, idx):
        return InputTarget(**super().__getitem__(idx))

    def __getitems__(self, idx):
        return InputTarget(**super().__getitems__(idx))

    @property
    def input(self) -> IdxDset:
        return self.dsets["input"]

    @property
    def target(self) -> IdxDset:
        return self.dsets["target"]

    @property
    def input_spec(self) -> rldata.TensorSpec:
        return self.input.__spec__()

    @property
    def target_spec(self) -> rldata.TensorSpec:
        return self.target.__spec__()
