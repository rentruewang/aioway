# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch.utils import data

from aioway.attrs import Attr, AttrDict
from aioway.hop import Hop, hop_dcls

__all__ = [
    "DataLoaderOpt",
    "DataLoaderHop",
    "Stream",
    "Sink",
    "Frame",
    "TensorStream",
    "TdictStream",
    "TensorFrame",
    "TdictFrame",
]


@dcls.dataclass(frozen=True)
class DataLoaderOpt:
    """
    The optoins for `data.DataLoader` on `Frame` in `FrameStream`.
    """

    batch_size: int = 1
    "The batch size of individual batches."

    drop_last: bool = False
    "Whether to drop the last batch, which may have a different `batch_size`."

    shuffle: bool = False
    "To shuffle or not."

    sampler: data.Sampler[int] | None = None
    "How to sample in case when want to shuffle."


@hop_dcls
class DataLoaderHop[T = typing.Any](Hop):
    """
    A `Hop` backed by a `torch` `DataLoader`.
    """

    dset: data.Dataset[T]
    """
    The data loader that would be iterated over.
    """

    opts: DataLoaderOpt
    """
    The options to pass to `DataLoader`.
    """

    def iterate(self) -> cabc.Generator[T]:
        loader = data.DataLoader(self.dset, **dcls.asdict(self.opts))
        yield from loader


class TensorAttrMixin(abc.ABC):
    """
    A `torch.Tensor` `Dataset` should also provide `.attr`.
    """

    @property
    @abc.abstractmethod
    def attr(self) -> Attr:
        raise NotImplementedError


class TdictAttrsMixin(abc.ABC):
    """
    A `td.TensorDict` `Dataset` should also provide `.attr`.
    """

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        raise NotImplementedError


class Stream[T = typing.Any](data.IterableDataset[T], abc.ABC):
    """
    `Stream` represents a set of sequential data stored somewhere.
    Each item is a single row of data.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __iter__(self) -> cabc.Iterator[T]:
        raise NotImplementedError

    def __call__(self, opts: DataLoaderOpt = DataLoaderOpt(), /) -> DataLoaderHop:
        return DataLoaderHop(self, opts)


class TensorStream(Stream[torch.Tensor], TensorAttrMixin, abc.ABC):
    """
    A `TensorStream` is a `Stream` of `torch.Tensor`s.
    """


class TdictStream(Stream[td.TensorDict], TdictAttrsMixin, abc.ABC):
    """
    A `TdictStream` is a `Stream` of `torch.Tensor`s.
    """


class Sink[T = typing.Any](abc.ABC):
    """
    Consumes a `Hop` and writes to some external location.
    """

    TYPE: typing.ClassVar[type[T]]
    """
    The type to check
    """

    def __call__(self, hop: Hop[T]) -> None:
        for batch in hop:
            if not isinstance(batch, self.TYPE):
                raise TypeError(f"The batch has {type(batch)=}, expected {self.TYPE}.")

            self.write(batch)

    @abc.abstractmethod
    def write(self, batch: T) -> None:
        raise NotImplementedError


class Frame[T = typing.Any](data.Dataset[T], abc.ABC):
    """
    `Frame` is a `Stream` that supports random access.
    Each item retrieved from `Frame` is a single row of data.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> T:
        """
        Get 1 item.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __getitems__(self, idx: list[int], /) -> T:
        """
        Get multiple items.
        """

        raise NotImplementedError

    def __call__(self, opts: DataLoaderOpt = DataLoaderOpt(), /) -> DataLoaderHop:
        return DataLoaderHop(self, opts)


class TensorFrame(Frame[torch.Tensor], TensorAttrMixin):
    """
    A `torch.Tensor` dataset that supports random access.
    """


class TdictFrame(Frame[torch.Tensor], TdictAttrsMixin):
    """
    A dataset of `td.TensorDict` that supports random access.
    """
