# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `Stream`."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._frames import TdictFrame, TensorFrame, frame_dcls
from aioway._streams import TdictStream, TensorStream, stream_dcls
from aioway.schemas import Attr, AttrDict

__all__ = ["StreamColumnView", "StreamSelectView", "FrameColumnView", "FrameSelectView"]


@stream_dcls
class StreamColumnView(TensorStream):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictStream
    "The input `TdictStream` to perform views on."

    col: str
    "The column to view on."

    @typing.override
    def read(self) -> torch.Tensor:
        tdict = next(self.input)
        return tdict[self.col]

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attr(self) -> Attr:
        return self.input.attrs[self.col]


@stream_dcls
class StreamSelectView(TdictStream):
    """
    The view generated when calling `Stream.select`.
    """

    input: TdictStream
    "The input `TdictStream` to perform views on."

    cols: cabc.Sequence[str]
    "The column to view on."

    def __iter__(self) -> typing.Self:
        return self

    @typing.override
    def read(self) -> td.TensorDict:
        batch = next(self.input)
        return batch.select(*self.cols)

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self.input.attrs.select(*self.cols)


@frame_dcls
class FrameColumnView(TensorFrame):
    """
    A column reference to a `Frame`.
    Performs `__getitem__` on a `Frame`, then select the column.
    """

    dset: TdictFrame
    "The frame to get column from"

    col: str
    "The column to view."

    def __len__(self) -> int:
        return len(self.dset)

    def __getitems__(self, idx):
        batch = self.dset.__getitems__(idx)
        return batch[self.col]

    @property
    @typing.override
    def attr(self) -> Attr:
        return self.dset.attrs[self.col]


@frame_dcls
class FrameSelectView(TdictFrame):
    """
    A selection view on the `Frame`.
    """

    dset: TdictFrame
    "The frame to get column from"

    cols: cabc.Sequence[str]
    "The column to view."

    @typing.override
    def __len__(self) -> int:
        return len(self.dset)

    def __getitems__(self, idx: list[int], /) -> td.TensorDict:
        items = self.dset.__getitems__(idx)
        return items.select(*self.cols)

    @property
    def attrs(self):
        return self.dset.attrs.select(*self.cols)
