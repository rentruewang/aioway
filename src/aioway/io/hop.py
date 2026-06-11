# Copyright (c) AIoWay Authors - All Rights Reserved

"`Frame`s that produce data by slicing contiguous input records."

import functools
import math
import typing

from torch.utils import data

from aioway.attrs import AttrDict
from aioway.hop import TdictHop, hop_dcls

from ._frames import TdictFrame

__all__ = ["FrameHopLoader", "FrameHop"]


@hop_dcls
class FrameHopLoader:
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
class FrameHop(TdictHop):
    """
    A `Stream` backed by a `Frame`.
    """

    frame: TdictFrame
    "The underlying `Frame`."

    options: FrameHopLoader
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
