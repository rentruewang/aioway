# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `Stream`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.dsets import DatasetColumnView, DatasetSelectView

from .streams import Stream

__all__ = ["StreamColumnView", "StreamSelectView"]


@dcls.dataclass(frozen=True)
class StreamColumnView(cabc.Iterator[torch.Tensor], DatasetColumnView[Stream]):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    def __iter__(self) -> typing.Self:
        return self

    def __next__(self) -> torch.Tensor:
        batch = next(self.dset)
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: Stream, /, column: str) -> typing.Self:
        return cls(col=column, dset=dataset)


@dcls.dataclass(frozen=True)
class StreamSelectView(DatasetSelectView[Stream], Stream):
    """
    The view generated when calling `Stream.select`.
    """

    COLUMN_TYPE = StreamColumnView

    def __iter__(self) -> typing.Self:
        return self

    @typing.override
    def _next(self) -> td.TensorDict:
        batch = next(self.dset)
        return batch.select(*self.cols)

    @property
    @typing.override
    def size(self) -> int:
        return self.dset.size

    @typing.override
    def _inputs(self):
        return (self.dset,)

    @classmethod
    def from_columns(cls, dataset: Stream, /, *columns: str) -> typing.Self:
        return cls(dset=dataset, cols=columns)
