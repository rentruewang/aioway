# Copyright (c) AIoWay Authors - All Rights Reserved

"`HopColumn`s are a column of `Hop`."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.hop import TdictHop, TensorHop, hop_dcls

__all__ = ["HopColumnView", "HopSelectView"]


@hop_dcls
class HopColumnView(TensorHop):
    """
    A column reference (on a `Hop`).
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


@hop_dcls
class HopSelectView(TdictHop):
    """
    The view generated when calling `Hop.select`.
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
