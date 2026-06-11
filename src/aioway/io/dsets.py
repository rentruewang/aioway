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
from aioway.hop import TdictHop, TensorHop

__all__ = [
    "LoaderOpt",
    "LoaderHop",
    "Stream",
    "Sink",
    "Frame",
    "dset_dcls",
    "TensorStream",
    "TdictStream",
    "TensorFrame",
    "TdictFrame",
]


@dcls.dataclass(frozen=True)
class LoaderOpt:
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
class LoaderHop[T = typing.Any](Hop[T]):
    """
    A `Hop` backed by a `torch` `DataLoader`.
    """

    _: dcls.KW_ONLY

    dset: data.Dataset[T]
    """
    The data loader that would be iterated over.
    """

    opts: LoaderOpt = LoaderOpt()
    """
    The options to pass to `DataLoader`.
    """

    def iterate(self) -> cabc.Generator[T]:
        loader = data.DataLoader(self.dset, **dcls.asdict(self.opts))
        yield from loader


@hop_dcls
class TensorLoaderHop(LoaderHop[torch.Tensor], TensorHop):
    """
    Load tensors.
    """

    schema: Attr
    "The attribute to check against."

    @property
    @typing.override
    def attr(self) -> Attr:
        return self.schema

    @typing.override
    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for batch in super().iterate():
            if Attr.parse(batch) != self.attr:
                raise ValueError(
                    f"The yielded {batch=} does not match the {self.attr=}."
                )

            yield batch


@hop_dcls
class TdictLoaderHop(LoaderHop[td.TensorDict], TdictHop):
    """
    Load tensors.
    """

    schema: AttrDict
    "The attribute to check against."

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self.schema

    @typing.override
    def iterate(self) -> cabc.Generator[td.TensorDict]:
        for batch in super().iterate():
            if AttrDict.parse(batch) != self.attrs:
                raise ValueError(
                    f"The yielded {batch=} does not match the {self.attrs=}."
                )

            yield batch


@typing.dataclass_transform()
def dset_dcls(cls):
    return dcls.dataclass(cls)


class TensorAttrMixin(abc.ABC):
    """
    A `torch.Tensor` `Dataset` should also provide `.attr`.
    """

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> TensorLoaderHop:
        return TensorLoaderHop(dset=self, opts=opts, schema=self.attr)

    @property
    @abc.abstractmethod
    def attr(self) -> Attr:
        raise NotImplementedError


class TdictAttrsMixin(abc.ABC):
    """
    A `td.TensorDict` `Dataset` should also provide `.attr`.
    """

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> TdictLoaderHop:
        return TdictLoaderHop(dset=self, opts=opts, schema=self.attrs)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        raise NotImplementedError


@dset_dcls
class Stream[T = typing.Any](data.IterableDataset[T], abc.ABC):
    """
    `Stream` represents a set of sequential data stored somewhere.
    Each item is a single row of data.
    """

    @abc.abstractmethod
    def __iter__(self) -> cabc.Iterator[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> LoaderHop:
        raise NotImplementedError


class TensorStream(TensorAttrMixin, Stream[torch.Tensor], abc.ABC):
    """
    A `TensorStream` is a `Stream` of `torch.Tensor`s.
    """


class TdictStream(TdictAttrsMixin, Stream[td.TensorDict], abc.ABC):
    """
    A `TdictStream` is a `Stream` of `torch.Tensor`s.
    """


@dset_dcls
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


@dset_dcls
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

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> LoaderHop:
        return LoaderHop(dset=self, opts=opts)


class TensorFrame(TensorAttrMixin, Frame[torch.Tensor], abc.ABC):
    """
    A `torch.Tensor` dataset that supports random access.
    """


class TdictFrame(TdictAttrsMixin, Frame[td.TensorDict], abc.ABC):
    """
    A dataset of `td.TensorDict` that supports random access.
    """
