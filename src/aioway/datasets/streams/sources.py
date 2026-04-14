# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Stream` that records past histories and supports random access."

import abc
import dataclasses as dcls
import functools
import logging
import math
import typing
from collections import abc as cabc

import tensordict as td
from torch.utils import data

from aioway._common import is_list_of
from aioway.schemas import AttrSet, attr_set

from ..frames import Frame
from .streams import Stream

__all__ = [
    "BoundedStream",
    "CacheStream",
    "ListStream",
    "FrameStream",
    "FrameStreamLoader",
]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class BoundedStream(Stream, abc.ABC):
    """
    A stream with `__len__` and `__getitem__`.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        "The number of batches saved in the current `Stream`."

        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._getitem_int(key)

        if isinstance(key, str) or is_list_of(str)(key):
            return Stream.__getitem__(self, key)

        raise TypeError(f"Do not know how to handle {type(key)=}.")

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> td.TensorDict:
        """
        Get individual items. Does not support slice input.

        Args:
            idx: An integer. Must be in the range `[-len(self), len(self))`.

        Returns:
            The `Chunk` batch.
        """


@dcls.dataclass(frozen=True)
class CacheStream(BoundedStream):
    """
    Exhaust the input stream, store it into a cache for repeating access.
    """

    stream: Stream
    "The input stream."

    saved: list[td.TensorDict] = dcls.field(default_factory=list)
    "The cache for the input `Stream`."

    @typing.override
    def __iter__(self) -> typing.Self:
        return dcls.replace(self, stream=self.stream, saved=self.saved)

    @typing.override
    def __len__(self) -> int:
        return len(self.saved)

    @typing.override
    def _getitem_int(self, idx):
        return self.saved[idx]

    @typing.override
    def _next(self) -> td.TensorDict:
        LOGGER.debug(
            "Executing `__iter__` for `CacheStream`. self.idx=%s, stream.idx=%s",
            self.idx,
            self.stream.idx,
        )

        if self.idx > self.stream.idx or self.idx > len(self.saved):
            raise AssertionError(
                "Invalid idx. Not synced properly with stream or cache."
            )

        # Try to get from `self.saved` first.
        if self.idx < len(self):
            return self._getitem_int(self.idx)

        # Now all the previous ones still must all have been saved.
        assert self.idx == len(self), f"{self.idx=} for {len(self)=}"

        # This shall raise `StopIteration` after done.
        # This may be fragile.
        item = next(self.stream)
        self.saved.append(item)
        return item

    @typing.override
    def _inputs(self):
        return (self.stream,)

    @property
    @typing.override
    def size(self) -> int:
        return self.stream.size

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.stream.attrs


@dcls.dataclass(frozen=True)
class ListStream(BoundedStream):
    "A `Stream` backed by a list of `TensorDict`."

    sequence: cabc.Sequence[td.TensorDict]
    "List of `td.TensorDict`s."

    @typing.override
    def __len__(self) -> int:
        return self.size

    @typing.override
    def _getitem_int(self, idx: int) -> td.TensorDict:
        return self.sequence[idx]

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self._schema

    @functools.cached_property
    def _schema(self) -> AttrSet:
        schemas = {attr_set(chunk) for chunk in self.sequence}

        if len(schemas) == 1:
            [schema] = schemas
            return schema

        raise ValueError("Chunks should have the same schema.")

    @typing.override
    def _next(self) -> td.TensorDict:
        if self.idx < self.size:
            return self._getitem_int(self.idx)
        else:
            raise StopIteration

    @typing.override
    def _inputs(self):
        return ()


@dcls.dataclass(frozen=True)
class FrameStreamLoader:
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


@dcls.dataclass(frozen=True)
class FrameStream(Stream):
    """
    A `Stream` backed by a `Frame`.
    """

    frame: Frame
    "The underlying `Frame`."

    options: FrameStreamLoader
    """
    The options passed directly to `data.DataLoader`.
    """

    @typing.override
    def _next(self) -> td.TensorDict:
        try:
            return self._get_batch(self.idx)
        except IndexError as ie:
            raise StopIteration from ie

    @functools.cached_property
    @typing.no_type_check
    def _dataloader(self) -> data.DataLoader:
        # Note that `__dict__` of a dataclass is just the custom fields.
        return data.DataLoader(
            self.frame,
            **self.options.__dict__,
            collate_fn=_identity,
        )

    @property
    @typing.override
    def size(self) -> int:
        return self._size

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.frame.attrs

    @functools.cached_property
    def _size(self) -> int:
        batch_size = self.options.batch_size
        drop_last = self.options.drop_last
        rounding = math.floor if drop_last else math.ceil
        return rounding(len(self.frame) / batch_size)

    def _get_batch(self, idx: int) -> td.TensorDict:
        batch_size = self.options.batch_size

        if not -self.size <= idx < self.size:
            raise IndexError(
                f"Index {idx=} out of bounds for stream {self.size=}, "
                f"backed by {self.frame}."
            )

        idx %= self.size
        return self.frame._getitems_batch(list(range(idx, idx + batch_size)))

    @typing.override
    def _inputs(self):
        return ()


def _identity[T](item: T) -> T:
    return item
