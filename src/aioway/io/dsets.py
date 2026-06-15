# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch.utils import data

from aioway._utils import HasLen
from aioway.hop import Hop, TdictHop, TensorHop, hop_dcls
from aioway.tags import TagDict

__all__ = [
    "LoaderOpt",
    "LoaderHop",
    "TensorLoaderHop",
    "TdictLoaderHop",
    "Dset",
    "Stream",
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

    @property
    @typing.override
    def size(self) -> int:
        # May be a frame or stream, so we need to test.
        # Perhaps this is a bad design.
        if not isinstance(self.dset, HasLen):
            return NotImplemented

        # Drop last or not would affect counts.
        count, remain = divmod(len(self.dset), self.opts.batch_size)

        if remain and not self.opts.drop_last:
            count += 1

        return count

    def iterate(self) -> cabc.Generator[T]:
        yield from self._dataloader()

    def _dataloader(self) -> data.DataLoader[T]:
        return data.DataLoader(self.dset, **dcls.asdict(self.opts))


@hop_dcls
class TensorLoaderHop(LoaderHop[torch.Tensor], TensorHop):
    """
    A `Hop` to load `torch.Tensor`.
    """


@hop_dcls
class TdictLoaderHop(LoaderHop[td.TensorDict], TdictHop):
    """
    A `Hop` to load `td.TensorDict`.
    """

    @typing.override
    def _dataloader(self) -> data.DataLoader[td.TensorDict]:
        return data.DataLoader(self.dset, collate_fn=td.stack, **dcls.asdict(self.opts))


@typing.dataclass_transform(frozen_default=True)
def dset_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


class _TensorAttrMixin(data.Dataset[torch.Tensor], metaclass=abc.ABCMeta):
    """
    A `torch.Tensor` `Dataset` should also provide `.attr`.
    """

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> TensorLoaderHop:
        # Set batch size to the ones provided.
        return TensorLoaderHop(dset=self, opts=opts)


class _TdictAttrsMixin(data.Dataset[td.TensorDict], metaclass=abc.ABCMeta):
    """
    A `td.TensorDict` `Dataset` should also provide `.attr`.
    """

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> TdictLoaderHop:
        return TdictLoaderHop(dset=self, opts=opts)


@dset_dcls
class Dset[T](data.Dataset[T]):
    """
    The base class for I/O.
    """

    @typing.final
    def __post_init__(self) -> None:
        self._setup()
        self._register()

    def __call__(self, opts: LoaderOpt = LoaderOpt(), /) -> LoaderHop:
        return LoaderHop(dset=self, opts=opts)

    def _setup(self) -> None:
        """
        Subclass should overwrite this function to perform setup. Can raise errors.
        """

    def _register(self) -> None:
        """
        Register the instance to a session.
        """

        from .sess import DsetSession

        if sess := DsetSession.current():
            sess.push(self)

    @property
    def tags(self) -> TagDict:
        """
        The tags associated with the data.
        """

        return TagDict()


@dset_dcls
class Stream[T = typing.Any](Dset[T], data.IterableDataset[T], abc.ABC):
    """
    `Stream` represents a set of sequential data stored somewhere.
    Each item is a single row of data.
    """

    @abc.abstractmethod
    def __iter__(self) -> cabc.Iterator[T]:
        raise NotImplementedError


class TensorStream(_TensorAttrMixin, Stream[torch.Tensor], abc.ABC):
    """
    A `TensorStream` is a `Stream` of `torch.Tensor`s.
    """


class TdictStream(_TdictAttrsMixin, Stream[td.TensorDict], abc.ABC):
    """
    A `TdictStream` is a `Stream` of `torch.Tensor`s.
    """


@dset_dcls
class Frame[T = typing.Any](Dset[T], data.Dataset[T], abc.ABC):
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


class TensorFrame(_TensorAttrMixin, Frame[torch.Tensor], abc.ABC):
    """
    A `torch.Tensor` dataset that supports random access.
    """


class TdictFrame(_TdictAttrsMixin, Frame[td.TensorDict], abc.ABC):
    """
    A dataset of `td.TensorDict` that supports random access.
    """
