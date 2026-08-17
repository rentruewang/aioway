# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch.utils import data as dutils

from aioway._utils import IntArray
from torchrl import data as rldata

__all__ = ["Dset", "IdxDset", "IterDset"]


class Dset[T](abc.ABC):
    """
    A `Dset` supports conversion to `dutils.Dataset`.

    It does not provide any additional functionality, but improves the API usage.
    """

    @abc.abstractmethod
    def __dataset__(self) -> dutils.Dataset[T]:
        """
        The dataset that this `Dset` converts to.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __spec__(self) -> rldata.TensorSpec:
        "The spec constraint for the data."

        raise NotImplementedError


class IdxDset[T](Dset[T], abc.ABC):
    """
    This is the type of `Dset` that is indexed.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int, /):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitems__(self, index: IntArray, /):
        raise NotImplementedError

    @typing.final
    def __dataset__(self) -> _IdxDataset:
        return _IdxDataset(self)


class _IdxDataset(dutils.Dataset):
    def __init__(self, dset: IdxDset) -> None:
        super().__init__()

        if not isinstance(dset, IdxDset):
            raise TypeError(f"{type(dset)=} should be `IdxDset`.")

        self._dset = dset

    def __repr__(self) -> str:
        return f"IdxDataset({self.dset!r})"

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index):
        return self.dset[index]

    def __getitems__(self, index):
        return self.dset.__getitems__(index)

    @property
    def dset(self) -> IdxDset:
        return self._dset


class IterDset[T](Dset[T], abc.ABC):
    """
    This is the iterator version of `Dset`.
    """

    @abc.abstractmethod
    def __iter__(self) -> cabc.Iterator[T]:
        """
        The method that yields the data in a sequential manner.
        """

        raise NotImplementedError


class _IterDataset(dutils.IterableDataset):
    def __init__(self, dset: IterDset) -> None:
        super().__init__()

        if not isinstance(dset, IterDset):
            raise TypeError(f"{type(dset)=} should be `IterDset`.")

        self._dset = dset

    def __repr__(self) -> str:
        return f"IterDataset({self.dset!r})"

    def __iter__(self):
        yield from self.dset

    @property
    def dset(self) -> IterDset:
        return self._dset
