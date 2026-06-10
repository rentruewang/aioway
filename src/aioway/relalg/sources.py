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

from aioway._frames import TdictFrame
from aioway.attrs import AttrDict
from aioway.hop import TdictHop, hop_dcls

__all__ = [
    "BoundedHop",
    "CacheStream",
    "ListStream",
    "FrameStream",
    "FrameStreamLoader",
]

LOGGER = logging.getLogger(__name__)


@hop_dcls
class BoundedHop(TdictHop, abc.ABC):
    """
    A stream with `__len__` and `__getitem__`.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        "The number of batches saved in the current `Stream`."

        raise NotImplementedError

    @typing.final
    def __getitem__(self, key):
        """
        Get individual items. Does not support slice input.

        Args:
            idx: An integer. Must be in the range `[-len(self), len(self))`.

        Returns:
            The `td.TensorDict` batch.
        """

        if isinstance(key, int):
            return self._getitem_int(key)

        raise TypeError(f"Do not know how to handle {type(key)=}.")

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> td.TensorDict:

        raise NotImplementedError


@hop_dcls
class CacheStream(BoundedHop):
    """
    Exhaust the input stream, store it into a cache for repeating access.
    """

    stream: TdictHop
    "The input stream."

    saved: list[td.TensorDict] = dcls.field(default_factory=list)
    "The cache for the input `Stream`."

    @typing.override
    def __len__(self) -> int:
        return len(self.saved)

    @typing.override
    def iterate(self):
        for batch in self.stream:
            self.saved.append(batch)
            yield batch

    @typing.override
    def _getitem_int(self, idx):
        return self.saved[idx]

    @property
    @typing.override
    def size(self) -> int:
        return self.stream.size

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self.stream.attrs


@hop_dcls
class ListStream(BoundedHop):
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
    def attrs(self) -> AttrDict:
        return self._schema

    @functools.cached_property
    def _schema(self) -> AttrDict:
        schemas = [AttrDict.parse(chunk) for chunk in self.sequence]

        if len({*schemas}) == 1:
            return schemas[0]

        raise ValueError("Chunks should have the same schema.")

    def iterate(self):
        for batch in self.sequence:
            yield batch


@hop_dcls
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


@hop_dcls
class FrameStream(TdictHop):
    """
    A `Stream` backed by a `Frame`.
    """

    frame: TdictFrame
    "The underlying `Frame`."

    options: FrameStreamLoader
    """
    The options passed directly to `data.DataLoader`.
    """

    @typing.override
    def iterate(self):
        for batch in self._dataloader:
            yield batch

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
    def attrs(self) -> AttrDict:
        return self.frame.attrs

    @functools.cached_property
    def _size(self) -> int:
        batch_size = self.options.batch_size
        drop_last = self.options.drop_last
        rounding = math.floor if drop_last else math.ceil
        return rounding(len(self.frame) / batch_size)


def _identity[T](item: T) -> T:
    return item
