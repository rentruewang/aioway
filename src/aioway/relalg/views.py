# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `Stream`."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.attrs import Attr, AttrDict
from aioway.hop import TdictHop, TensorHop, hop_dcls

__all__ = ["StreamColumnView", "StreamSelectView", "FrameColumnView", "FrameSelectView"]


@hop_dcls
class StreamColumnView(TensorHop):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictHop
    "The input `TdictHop` to perform views on."

    col: str
    "The column to view on."

    @typing.override
    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for batch in self.input:
            result = batch[self.col]
            assert isinstance(result, torch.Tensor)
            yield result

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attr(self) -> Attr:
        return self.input.attrs[self.col]


@hop_dcls
class StreamSelectView(TdictHop):
    """
    The view generated when calling `Stream.select`.
    """

    input: TdictHop
    "The input `TdictHop` to perform views on."

    cols: cabc.Sequence[str]
    "The column to view on."

    @typing.override
    def iterate(self) -> cabc.Generator[td.TensorDict]:
        for batch in self.input:
            yield batch.select(*self.cols)

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self.input.attrs.select(*self.cols)
