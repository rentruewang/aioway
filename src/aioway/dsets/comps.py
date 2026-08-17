# Copyright (c) AIoWay Authors - All Rights Reserved

"A module for composite `Dset`s."

import abc
import typing
from collections import abc as cabc

import tensordict as td
from torchrl import data as rldata

from .dsets import Dset, IdxDset

__all__ = ["CompositeDset", "ComposedDset", "IdxComposedDset"]


class CompositeDset(Dset, abc.ABC):
    """
    A `Dset` that the data should be composites.
    """

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
