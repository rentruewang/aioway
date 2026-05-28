# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `Stream`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._streams import TdictStream, TensorStream
from aioway.schemas import Attr, AttrDict

__all__ = ["StreamColumnView", "StreamSelectView"]


@dcls.dataclass(frozen=True)
class StreamColumnView(TensorStream):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictStream
    "The input `TdictStream` to perform views on."

    column: str
    "The column to view on."

    @typing.override
    def read(self) -> torch.Tensor:
        tdict = next(self.input)
        return tdict[self.column]

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attr(self) -> Attr:
        return self.input.attrs[self.column]


@dcls.dataclass(frozen=True)
class StreamSelectView(TdictStream):
    """
    The view generated when calling `Stream.select`.
    """

    input: TdictStream
    "The input `TdictStream` to perform views on."

    columns: cabc.Sequence[str]
    "The column to view on."

    def __iter__(self) -> typing.Self:
        return self

    @typing.override
    def read(self) -> td.TensorDict:
        batch = next(self.input)
        return batch.select(*self.columns)

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self.input.attrs.select(*self.columns)
