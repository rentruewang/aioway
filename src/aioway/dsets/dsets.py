# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import numpy as np
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs

from aioway._utils import IntArray

__all__ = ["Dset", "IdxDset", "IterDset", "DatasetIdxDset"]


class Dset[T](abc.ABC):
    """
    A `Dset` supports conversion to `dutils.Dataset`.

    It does not provide any additional functionality, but improves the API usage.
    """

    @abc.abstractmethod
    def __tspec__(self) -> tspecs.TensorSpec:
        "The spec constraint for the data."

        raise NotImplementedError

    @property
    def collate_fn(self) -> cabc.Callable[..., typing.Any] | None:
        """
        Overwrite this function if `__getitems__` gives an output
        that cannot be handled by default collate.
        """

        return None

    @abc.abstractmethod
    def to_dataset(self) -> dutils.Dataset[T]:
        """
        The dataset that this `Dset` converts to.
        """

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

    @typing.override
    def to_dataset(self):
        return _IdxDataset(self)


class DatasetIdxDset[T](IdxDset[T]):
    """
    Adaptor from `dutils.Dataset` to `IdxDset`.
    """

    def __init__(
        self, dataset: dutils.Dataset[T], spec: tspecs.TensorSpec, collate_fn=None
    ) -> None:
        self._dataset = dataset
        self._spec = spec
        self._colalte_fn = collate_fn

    @typing.no_type_check
    def __len__(self) -> int:
        return len(self._dataset)

    @typing.no_type_check
    def __getitem__(self, idx):
        return self._dataset[idx]

    @typing.no_type_check
    def __getitems__(self, idx):
        return self._dataset.__getitems__(idx)

    def to_dataset(self):
        return self

    def __tspec__(self) -> tspecs.TensorSpec:
        return self._spec

    @property
    def collate_fn(self):
        return self._colalte_fn

    @property
    def dataset(self) -> dutils.Dataset[T]:
        return self._dataset


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
        return self.dset.__getitems__(np.asarray(index))

    def __tspec__(self) -> tspecs.TensorSpec:
        return self.dset.__tspec__()

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

    @typing.override
    def to_dataset(self):
        return _IterDataset(self)


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

    def __tspec__(self) -> tspecs.TensorSpec:
        return self.dset.__tspec__()

    @property
    def dset(self) -> IterDset:
        return self._dset
